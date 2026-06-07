"""
content_planner/providers.py

Domain Layer - Data classes, enums, and business logic for content planning.

Contains:
- Enums: SectionType, ContentType, SourcePreference, DocumentType, etc.
- Data classes: Section, StructuredPlan, Microprompt, Template, etc.
- Managers: TemplateManager, MicropromptEngine, PlanGenerator, Validators

Version: 1.0.0
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class SectionType(str, Enum):
    """Types of document sections."""
    TITLE = "title"
    EXECUTIVE_SUMMARY = "executive_summary"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    FINDINGS = "findings"
    DATA = "data"
    DISCUSSION = "discussion"
    RECOMMENDATIONS = "recommendations"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    REFERENCES = "references"
    CUSTOM = "custom"


class ContentType(str, Enum):
    """Type of content for a section."""
    PROSE = "prose"
    TABLE = "table"
    CHART = "chart"
    LIST = "list"
    MIXED = "mixed"
    DATA_DRIVEN = "data_driven"


class SourcePreference(str, Enum):
    """Source preference for research."""
    RAG_ONLY = "rag_only"
    WEB_ONLY = "web_only"
    RAG_FIRST = "rag_first"
    WEB_FIRST = "web_first"
    MIXED = "mixed"
    LLM_REASONING = "llm_reasoning"
    ADAPTIVE = "adaptive"


class DocumentType(str, Enum):
    """Type of document being planned."""
    REPORT = "report"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    PRESENTATION = "presentation"
    BRIEF = "brief"
    PROPOSAL = "proposal"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    CUSTOM = "custom"


class FormalityLevel(str, Enum):
    """Formality level of the document."""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    LEGAL = "legal"


class WritingStyle(str, Enum):
    """Writing style for content."""
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    PERSUASIVE = "persuasive"
    ANALYTICAL = "analytical"
    DESCRIPTIVE = "descriptive"
    INSTRUCTIONAL = "instructional"


class Tone(str, Enum):
    """Tone of the content."""
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    OPTIMISTIC = "optimistic"
    CRITICAL = "critical"


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================================
# Core Data Classes
# ============================================================================


@dataclass
class StyleGuide:
    """Style guidelines for content."""
    formality: FormalityLevel = FormalityLevel.PROFESSIONAL
    writing_style: WritingStyle = WritingStyle.FORMAL
    tone: Tone = Tone.NEUTRAL
    language: str = "it"
    use_active_voice: bool = True
    max_sentence_length: int = 30
    avoid_jargon: bool = False
    citation_style: str = "apa"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formality": self.formality.value,
            "writing_style": self.writing_style.value,
            "tone": self.tone.value,
            "language": self.language,
            "citation_style": self.citation_style,
        }


@dataclass
class ResearchConstraints:
    """Constraints for research phase."""
    min_sources: int = 3
    max_sources: int = 10
    source_preference: SourcePreference = SourcePreference.RAG_FIRST
    require_recent: bool = False
    recency_days: int = 365
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    require_authoritative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_sources": self.min_sources,
            "max_sources": self.max_sources,
            "source_preference": self.source_preference.value,
            "require_recent": self.require_recent,
        }


@dataclass
class ContentConstraints:
    """Constraints for content generation."""
    min_words: int = 100
    max_words: int = 1000
    min_paragraphs: int = 1
    max_paragraphs: int = 10
    require_citations: bool = True  # Validation constraint (enforce citations)
    require_data: bool = False
    require_examples: bool = False
    allow_speculation: bool = False
    # Bug 1 Fix: Add missing fields used by adapter
    language: str = "it"
    formality_level: Optional[FormalityLevel] = None
    max_sections: int = 15
    max_tokens_total: int = 15000
    max_tokens_per_section: int = 3000
    min_sections: int = 2
    target_audience: str = "general"
    include_citations: bool = True  # Generation option (add citations to output)
    include_toc: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_words": self.min_words,
            "max_words": self.max_words,
            "require_citations": self.require_citations,
            "language": self.language,
            "formality_level": self.formality_level.value if self.formality_level else None,
            "max_sections": self.max_sections,
            "max_tokens_total": self.max_tokens_total,
            "max_tokens_per_section": self.max_tokens_per_section,
            "min_sections": self.min_sections,
            "target_audience": self.target_audience,
            "include_citations": self.include_citations,
            "include_toc": self.include_toc,
        }


@dataclass
class Microprompt:
    """Microprompt for section generation."""
    section_id: str
    prompt: str
    system_context: str = ""
    output_format: str = "prose"
    constraints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def render(self, context: Dict[str, Any]) -> str:
        """Render microprompt with context variables."""
        rendered = self.prompt
        all_vars = {**self.variables, **context}
        for key, value in all_vars.items():
            rendered = rendered.replace(f"${{{key}}}", str(value))
            rendered = rendered.replace(f"${key}", str(value))
        return rendered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "prompt": self.prompt,
            "system_context": self.system_context,
            "output_format": self.output_format,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
        }


@dataclass
class Section:
    """A section in the document plan."""
    id: str
    title: str
    section_type: SectionType = SectionType.CUSTOM
    content_type: ContentType = ContentType.PROSE
    source_preference: SourcePreference = SourcePreference.RAG_FIRST
    description: str = ""
    order: int = 0
    depth: int = 1
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    research_query: str = ""
    research_constraints: Optional[ResearchConstraints] = None
    content_constraints: Optional[ContentConstraints] = None
    microprompt: Optional[Microprompt] = None
    dependencies: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)
    target_tokens: int = 500
    min_tokens: int = 100
    max_tokens: int = 1500
    required: bool = True
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "section_type": self.section_type.value,
            "content_type": self.content_type.value,
            "description": self.description,
            "order": self.order,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children": self.children,
            "research_query": self.research_query,
            "dependencies": self.dependencies,
            "microprompt": self.microprompt.to_dict() if self.microprompt else None,
        }


@dataclass
class StructuredPlan:
    """Complete document plan."""
    id: str
    title: str
    document_type: DocumentType = DocumentType.REPORT
    topic: str = ""
    description: str = ""
    sections: List[Section] = field(default_factory=list)
    style_guide: Optional[StyleGuide] = None
    global_constraints: Optional[ContentConstraints] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    estimated_tokens: int = 0
    estimated_sections: int = 0
    language: str = "it"
    constraints: Optional[ContentConstraints] = None

    @property
    def enabled_sections(self) -> List[Section]:
        """Return sections that are enabled (or all if no enabled flag)."""
        return [
            s for s in self.sections
            if getattr(s, "enabled", True)
        ]

    def get_section(self, section_id: str) -> Optional[Section]:
        """Get section by ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    def get_section_queries(self) -> List[Dict[str, Any]]:
        """Get research queries for all sections."""
        queries = []
        for section in self.sections:
            if section.research_query:
                queries.append({
                    "section_id": section.id,
                    "query": section.research_query,
                    "constraints": section.research_constraints.to_dict() if section.research_constraints else {},
                })
        return queries

    def get_execution_order(self) -> List[str]:
        """Get sections in dependency-aware execution order."""
        executed = set()
        order = []
        remaining = {s.id: s for s in self.sections}

        while remaining:
            ready = [
                sid for sid, s in remaining.items()
                if all(dep in executed for dep in s.dependencies)
            ]
            if not ready:
                ready = list(remaining.keys())[:1]
            for sid in sorted(ready, key=lambda x: remaining[x].order):
                order.append(sid)
                executed.add(sid)
                del remaining[sid]

        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "document_type": self.document_type.value,
            "topic": self.topic,
            "description": self.description,
            "sections": [s.to_dict() for s in self.sections],
            "style_guide": self.style_guide.to_dict() if self.style_guide else None,
            "section_count": len(self.sections),
            "version": self.version,
        }


# ============================================================================
# Template Classes
# ============================================================================


@dataclass
class TemplateSection:
    """Template for a section."""
    section_type: SectionType
    title_template: str
    description_template: str = ""
    content_type: ContentType = ContentType.PROSE
    required: bool = False
    order: int = 0
    default_constraints: Optional[ContentConstraints] = None
    microprompt_template: str = ""


@dataclass
class DocumentTemplate:
    """Template for a document type."""
    id: str
    name: str
    document_type: DocumentType
    description: str = ""
    sections: List[TemplateSection] = field(default_factory=list)
    style_guide: Optional[StyleGuide] = None
    variables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "document_type": self.document_type.value,
            "description": self.description,
            "section_count": len(self.sections),
            "tags": self.tags,
        }


# ============================================================================
# Validation Classes
# ============================================================================


@dataclass
class ValidationIssue:
    """A validation issue."""
    severity: ValidationSeverity
    message: str
    field: str = ""
    section_id: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "section_id": self.section_id,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of plan validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


# ============================================================================
# Template Manager
# ============================================================================


class TemplateManager:
    """Manages document templates."""

    # Built-in templates
    BUILTIN_TEMPLATES = {
        "report_standard": DocumentTemplate(
            id="report_standard",
            name="Standard Report",
            document_type=DocumentType.REPORT,
            description="Standard report with executive summary, analysis, and recommendations",
            sections=[
                TemplateSection(SectionType.EXECUTIVE_SUMMARY, "Executive Summary", required=True, order=1),
                TemplateSection(SectionType.INTRODUCTION, "Introduction", required=True, order=2),
                TemplateSection(SectionType.BACKGROUND, "Background", order=3),
                TemplateSection(SectionType.METHODOLOGY, "Methodology", order=4),
                TemplateSection(SectionType.FINDINGS, "Findings", required=True, order=5),
                TemplateSection(SectionType.ANALYSIS, "Analysis", order=6),
                TemplateSection(SectionType.RECOMMENDATIONS, "Recommendations", order=7),
                TemplateSection(SectionType.CONCLUSION, "Conclusion", required=True, order=8),
                TemplateSection(SectionType.REFERENCES, "References", order=9),
            ],
            tags=["business", "formal"],
        ),
        "presentation": DocumentTemplate(
            id="presentation",
            name="Presentation",
            document_type=DocumentType.PRESENTATION,
            description="Slide presentation format",
            sections=[
                TemplateSection(SectionType.TITLE, "Title Slide", required=True, order=1),
                TemplateSection(SectionType.INTRODUCTION, "Agenda", order=2),
                TemplateSection(SectionType.BACKGROUND, "Context", order=3),
                TemplateSection(SectionType.FINDINGS, "Key Points", required=True, order=4),
                TemplateSection(SectionType.CONCLUSION, "Summary", order=5),
            ],
            tags=["slides", "visual"],
        ),
        "technical_report": DocumentTemplate(
            id="technical_report",
            name="Technical Report",
            document_type=DocumentType.TECHNICAL,
            description="Technical documentation with detailed methodology",
            sections=[
                TemplateSection(SectionType.EXECUTIVE_SUMMARY, "Abstract", required=True, order=1),
                TemplateSection(SectionType.INTRODUCTION, "Introduction", required=True, order=2),
                TemplateSection(SectionType.BACKGROUND, "Literature Review", order=3),
                TemplateSection(SectionType.METHODOLOGY, "Methodology", required=True, order=4),
                TemplateSection(SectionType.DATA, "Data & Results", order=5, content_type=ContentType.DATA_DRIVEN),
                TemplateSection(SectionType.ANALYSIS, "Analysis", order=6),
                TemplateSection(SectionType.DISCUSSION, "Discussion", order=7),
                TemplateSection(SectionType.CONCLUSION, "Conclusion", required=True, order=8),
                TemplateSection(SectionType.REFERENCES, "References", order=9),
                TemplateSection(SectionType.APPENDIX, "Appendices", order=10),
            ],
            tags=["technical", "academic"],
        ),
        "analysis": DocumentTemplate(
            id="analysis",
            name="Analysis Report",
            document_type=DocumentType.ANALYSIS,
            description="Analytical report focused on data interpretation",
            sections=[
                TemplateSection(SectionType.EXECUTIVE_SUMMARY, "Executive Summary", required=True, order=1),
                TemplateSection(SectionType.INTRODUCTION, "Introduction", order=2),
                TemplateSection(SectionType.DATA, "Data Overview", content_type=ContentType.DATA_DRIVEN, order=3),
                TemplateSection(SectionType.ANALYSIS, "Analysis", required=True, order=4),
                TemplateSection(SectionType.FINDINGS, "Key Findings", order=5),
                TemplateSection(SectionType.RECOMMENDATIONS, "Recommendations", order=6),
                TemplateSection(SectionType.CONCLUSION, "Conclusion", order=7),
            ],
            tags=["analysis", "data"],
        ),
    }

    def __init__(self, templates_path: Optional[Path] = None):
        self.templates_path = templates_path
        self._templates: Dict[str, DocumentTemplate] = dict(self.BUILTIN_TEMPLATES)
        self._custom_templates: Dict[str, DocumentTemplate] = {}
        if templates_path:
            self._load_custom_templates()

    def get_template(self, template_id: str) -> Optional[DocumentTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id) or self._custom_templates.get(template_id)

    def list_templates(
        self,
        document_type: Optional[DocumentType] = None,
        category: Optional[str] = None,
    ) -> List[DocumentTemplate]:
        """List available templates, optionally filtered by type or category tag."""
        all_templates = list(self._templates.values()) + list(self._custom_templates.values())
        if document_type:
            all_templates = [t for t in all_templates if t.document_type == document_type]
        if category:
            all_templates = [t for t in all_templates if category in t.tags]
        return all_templates

    def get_categories(self) -> List[str]:
        """Return deduplicated list of all template tags (categories)."""
        tags: set[str] = set()
        for t in list(self._templates.values()) + list(self._custom_templates.values()):
            tags.update(t.tags)
        return sorted(tags)

    def register_template(self, template: DocumentTemplate) -> None:
        """Register a custom template."""
        self._custom_templates[template.id] = template

    def suggest_template(self, topic: str, document_type: Optional[DocumentType] = None) -> Optional[DocumentTemplate]:
        """Suggest best template for topic."""
        candidates = self.list_templates(document_type)
        if not candidates:
            return self._templates.get("report_standard")
        topic_lower = topic.lower()
        for template in candidates:
            for tag in template.tags:
                if tag in topic_lower:
                    return template
        return candidates[0]

    def _load_custom_templates(self) -> None:
        """Load custom templates from path."""
        if not self.templates_path or not self.templates_path.exists():
            return
        for file in self.templates_path.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                template = self._parse_template(data)
                self._custom_templates[template.id] = template
            except Exception as e:
                logger.warning(f"Failed to load template {file}: {e}")

    def _parse_template(self, data: Dict[str, Any]) -> DocumentTemplate:
        """Parse template from dict."""
        sections = []
        for s in data.get("sections", []):
            sections.append(TemplateSection(
                section_type=SectionType(s.get("type", "custom")),
                title_template=s.get("title", ""),
                description_template=s.get("description", ""),
                content_type=ContentType(s.get("content_type", "prose")),
                required=s.get("required", False),
                order=s.get("order", 0),
            ))
        return DocumentTemplate(
            id=data.get("id", ""),
            name=data.get("name", ""),
            document_type=DocumentType(data.get("document_type", "report")),
            description=data.get("description", ""),
            sections=sections,
            tags=data.get("tags", []),
        )


# ============================================================================
# Microprompt Engine
# ============================================================================


class MicropromptEngine:
    """Generates microprompts for sections."""

    # Section-specific prompt templates
    SECTION_PROMPTS = {
        SectionType.EXECUTIVE_SUMMARY: """Write an executive summary for a {document_type} about "{topic}".

Key requirements:
- Concise overview (max {max_words} words)
- Highlight key findings and recommendations
- Written for busy executives
- No technical jargon unless necessary

Context from research:
{context}

Style: {style}""",

        SectionType.INTRODUCTION: """Write an introduction for a {document_type} about "{topic}".

Requirements:
- Establish context and relevance
- State the purpose clearly
- Preview the structure
- Engage the reader

Context:
{context}

Style: {style}""",

        SectionType.METHODOLOGY: """Describe the methodology for this {document_type} about "{topic}".

Include:
- Approach and methods used
- Data sources
- Analysis techniques
- Limitations

Context:
{context}

Style: {style}""",

        SectionType.FINDINGS: """Present the key findings for this {document_type} about "{topic}".

Requirements:
- Clear, evidence-based statements
- Organized logically
- Supported by data/sources
- Objective tone

Research findings:
{context}

Style: {style}""",

        SectionType.ANALYSIS: """Provide analysis for this {document_type} about "{topic}".

Focus on:
- Interpretation of findings
- Patterns and trends
- Implications
- Critical evaluation

Data and context:
{context}

Style: {style}""",

        SectionType.RECOMMENDATIONS: """Write recommendations based on this {document_type} about "{topic}".

Requirements:
- Actionable suggestions
- Prioritized by importance
- Justified by findings
- Realistic and practical

Based on analysis:
{context}

Style: {style}""",

        SectionType.CONCLUSION: """Write a conclusion for this {document_type} about "{topic}".

Include:
- Summary of key points
- Main takeaways
- Future outlook (if relevant)
- Closing statement

Context:
{context}

Style: {style}""",

        SectionType.CUSTOM: """Write the "{section_title}" section for a {document_type} about "{topic}".

Section description: {section_description}

Requirements:
{constraints}

Context:
{context}

Style: {style}""",
    }

    def __init__(self):
        self._custom_prompts: Dict[str, str] = {}

    def generate_microprompt(
        self,
        section: Section,
        plan: StructuredPlan,
        context: str = "",
    ) -> Microprompt:
        """Generate microprompt for a section."""
        template = self._get_template(section.section_type)
        style_str = self._format_style(plan.style_guide)
        constraints_str = self._format_constraints(section.content_constraints)
        
        prompt = template.format(
            document_type=plan.document_type.value,
            topic=plan.topic,
            section_title=section.title,
            section_description=section.description,
            max_words=section.content_constraints.max_words if section.content_constraints else 500,
            context=context or "[Research context will be inserted here]",
            style=style_str,
            constraints=constraints_str,
        )

        return Microprompt(
            section_id=section.id,
            prompt=prompt,
            system_context=self._get_system_context(plan),
            output_format=section.content_type.value,
            constraints=constraints_str.split("\n") if constraints_str else [],
            dependencies=section.dependencies,
        )

    def _get_template(self, section_type: SectionType) -> str:
        """Get prompt template for section type."""
        if section_type in self._custom_prompts:
            return self._custom_prompts[section_type]
        return self.SECTION_PROMPTS.get(section_type, self.SECTION_PROMPTS[SectionType.CUSTOM])

    def _format_style(self, style: Optional[StyleGuide]) -> str:
        """Format style guide for prompt."""
        if not style:
            return "Professional, formal tone"
        return f"{style.writing_style.value.title()} style, {style.tone.value} tone, {style.formality.value} formality"

    def _format_constraints(self, constraints: Optional[ContentConstraints]) -> str:
        """Format constraints for prompt."""
        if not constraints:
            return ""
        parts = []
        parts.append(f"- Word count: {constraints.min_words}-{constraints.max_words}")
        if constraints.require_citations:
            parts.append("- Include citations")
        if constraints.require_data:
            parts.append("- Include supporting data")
        if constraints.require_examples:
            parts.append("- Include examples")
        return "\n".join(parts)

    def _get_system_context(self, plan: StructuredPlan) -> str:
        """Get system context for generation."""
        return f"""You are writing a {plan.document_type.value} about "{plan.topic}".
Document title: {plan.title}
Language: {plan.style_guide.language if plan.style_guide else 'it'}
Follow the style guide and constraints provided."""


# ============================================================================
# Plan Generator
# ============================================================================


class PlanGenerator:
    """Generates document plans from topics/requirements."""

    def __init__(
        self,
        template_manager: TemplateManager,
        microprompt_engine: MicropromptEngine,
    ):
        self.template_manager = template_manager
        self.microprompt_engine = microprompt_engine

    def generate_plan(
        self,
        topic: str,
        document_type: Optional[str] = None,
        template_id: Optional[str] = None,
        requirements: Optional[Dict[str, Any]] = None,
        style_guide: Optional[Dict[str, Any]] = None,
    ) -> StructuredPlan:
        """Generate a document plan."""
        requirements = requirements or {}
        doc_type = DocumentType(document_type) if document_type else DocumentType.REPORT

        # Get template
        if template_id:
            template = self.template_manager.get_template(template_id)
        else:
            template = self.template_manager.suggest_template(topic, doc_type)

        # Create plan
        plan = StructuredPlan(
            id=self._generate_id(topic),
            title=requirements.get("title", self._generate_title(topic, doc_type)),
            document_type=doc_type,
            topic=topic,
            description=requirements.get("description", ""),
            style_guide=self._create_style_guide(style_guide),
        )

        # Generate sections from template
        if template:
            plan.sections = self._sections_from_template(template, topic, requirements)
        else:
            plan.sections = self._generate_default_sections(topic, doc_type, requirements)

        # Generate microprompts
        for section in plan.sections:
            section.microprompt = self.microprompt_engine.generate_microprompt(section, plan)

        return plan

    def _generate_id(self, topic: str) -> str:
        """Generate plan ID."""
        hash_input = f"{topic}{datetime.utcnow().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _generate_title(self, topic: str, doc_type: DocumentType) -> str:
        """Generate document title."""
        type_prefix = {
            DocumentType.REPORT: "Report:",
            DocumentType.ANALYSIS: "Analysis:",
            DocumentType.RESEARCH: "Research:",
            DocumentType.PRESENTATION: "",
            DocumentType.BRIEF: "Brief:",
            DocumentType.PROPOSAL: "Proposal:",
            DocumentType.TECHNICAL: "Technical Report:",
        }
        prefix = type_prefix.get(doc_type, "")
        return f"{prefix} {topic}".strip()

    def _create_style_guide(self, style_config: Optional[Dict[str, Any]]) -> StyleGuide:
        """Create style guide from config."""
        if not style_config:
            return StyleGuide()
        return StyleGuide(
            formality=FormalityLevel(style_config.get("formality", "professional")),
            writing_style=WritingStyle(style_config.get("writing_style", "formal")),
            tone=Tone(style_config.get("tone", "neutral")),
            language=style_config.get("language", "it"),
            citation_style=style_config.get("citation_style", "apa"),
        )

    def _sections_from_template(
        self,
        template: DocumentTemplate,
        topic: str,
        requirements: Dict[str, Any],
    ) -> List[Section]:
        """Create sections from template."""
        sections = []
        for i, ts in enumerate(template.sections):
            section = Section(
                id=f"section_{i+1}",
                title=ts.title_template.format(topic=topic) if "{topic}" in ts.title_template else ts.title_template,
                section_type=ts.section_type,
                content_type=ts.content_type,
                description=ts.description_template,
                order=ts.order or i + 1,
                research_query=self._generate_research_query(topic, ts),
                content_constraints=ts.default_constraints or ContentConstraints(),
            )
            sections.append(section)
        return sections

    def _generate_default_sections(
        self,
        topic: str,
        doc_type: DocumentType,
        requirements: Dict[str, Any],
    ) -> List[Section]:
        """Generate default sections when no template."""
        default_types = [
            (SectionType.INTRODUCTION, "Introduction"),
            (SectionType.BACKGROUND, "Background"),
            (SectionType.ANALYSIS, "Analysis"),
            (SectionType.CONCLUSION, "Conclusion"),
        ]
        sections = []
        for i, (stype, title) in enumerate(default_types):
            sections.append(Section(
                id=f"section_{i+1}",
                title=title,
                section_type=stype,
                order=i + 1,
                research_query=f"{topic} {title.lower()}",
            ))
        return sections

    def _generate_research_query(self, topic: str, template_section: TemplateSection) -> str:
        """Generate research query for section."""
        type_queries = {
            SectionType.EXECUTIVE_SUMMARY: f"{topic} summary overview key points",
            SectionType.INTRODUCTION: f"{topic} introduction context background",
            SectionType.BACKGROUND: f"{topic} history background context",
            SectionType.METHODOLOGY: f"{topic} methodology approach methods",
            SectionType.FINDINGS: f"{topic} findings results data",
            SectionType.ANALYSIS: f"{topic} analysis interpretation trends",
            SectionType.RECOMMENDATIONS: f"{topic} recommendations solutions best practices",
            SectionType.CONCLUSION: f"{topic} conclusion summary implications",
        }
        return type_queries.get(template_section.section_type, topic)

    def estimate_tokens(self, plan) -> "PlanTokenEstimate":
        """
        Estimate token usage for a structured plan.

        Uses section word constraints to approximate token counts
        (~1.3 tokens per word for typical content).
        """
        tokens_per_word = 1.3
        default_max_words = 1000
        default_budget = 128000

        tokens_by_section = {}
        total_tokens = 0

        sections = plan.sections if hasattr(plan, "sections") else []
        for section in sections:
            max_words = default_max_words
            if hasattr(section, "content_constraints") and section.content_constraints:
                max_words = getattr(section.content_constraints, "max_words", default_max_words)
            section_tokens = int(max_words * tokens_per_word)
            sid = getattr(section, "id", f"section_{sections.index(section)}")
            tokens_by_section[sid] = section_tokens
            total_tokens += section_tokens

        budget = default_budget
        if hasattr(plan, "metadata") and isinstance(plan.metadata, dict):
            budget = plan.metadata.get("token_budget", default_budget)

        return PlanTokenEstimate(
            total_tokens=total_tokens,
            tokens_by_section=tokens_by_section,
            within_budget=total_tokens <= budget,
            budget_utilization=round(total_tokens / budget, 4) if budget > 0 else 0.0,
        )

    def estimate_resources(self, plan) -> "PlanResourceEstimate":
        """
        Estimate resources needed for plan execution.

        Calculates time, API calls, parallelism, and research breakdown
        based on plan sections.
        """
        sections = plan.sections if hasattr(plan, "sections") else []
        research_sections = [
            s for s in sections
            if getattr(s, "research_query", "")
        ]
        token_est = self.estimate_tokens(plan)
        return PlanResourceEstimate(
            estimated_time_minutes=round(len(sections) * 0.5, 1),
            estimated_api_calls=len(sections) + len(research_sections),
            parallel_batches=max(1, len(sections) // 3),
            sections_requiring_research=len(research_sections),
            sections_llm_only=len(sections) - len(research_sections),
            estimated_tokens=token_est,
        )


@dataclass
class PlanTokenEstimate:
    """Result of token estimation for a plan."""
    total_tokens: int = 0
    tokens_by_section: Dict[str, int] = field(default_factory=dict)
    within_budget: bool = True
    budget_utilization: float = 0.0


@dataclass
class PlanResourceEstimate:
    """Extended resource estimate for plan execution."""
    estimated_time_minutes: float = 0.0
    estimated_api_calls: int = 0
    parallel_batches: int = 1
    sections_requiring_research: int = 0
    sections_llm_only: int = 0
    estimated_tokens: PlanTokenEstimate = field(default_factory=PlanTokenEstimate)


# ============================================================================
# Validators
# ============================================================================


class PlanValidator:
    """Validates document plans."""

    def validate(self, plan: StructuredPlan) -> ValidationResult:
        """Validate a plan."""
        issues = []

        # Check required fields
        if not plan.title:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Plan title is required",
                field="title",
            ))

        if not plan.topic:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Topic is recommended",
                field="topic",
            ))

        if not plan.sections:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Plan must have at least one section",
                field="sections",
            ))

        # Validate sections
        section_ids = set()
        for section in plan.sections:
            section_issues = self._validate_section(section, section_ids)
            issues.extend(section_issues)
            section_ids.add(section.id)

        # Check dependencies
        dep_issues = self._validate_dependencies(plan)
        issues.extend(dep_issues)

        # Calculate score
        errors = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warnings = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        score = max(0, 1.0 - (errors * 0.2) - (warnings * 0.05))

        return ValidationResult(
            valid=errors == 0,
            issues=issues,
            score=score,
        )

    def _validate_section(self, section: Section, existing_ids: Set[str]) -> List[ValidationIssue]:
        """Validate a section."""
        issues = []

        if not section.id:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Section ID is required",
                field="id",
                section_id=section.id,
            ))

        if section.id in existing_ids:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Duplicate section ID: {section.id}",
                field="id",
                section_id=section.id,
            ))

        if not section.title:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Section title is recommended",
                field="title",
                section_id=section.id,
            ))

        return issues

    def _validate_dependencies(self, plan: StructuredPlan) -> List[ValidationIssue]:
        """Validate section dependencies."""
        issues = []
        section_ids = {s.id for s in plan.sections}

        for section in plan.sections:
            for dep in section.dependencies:
                if dep not in section_ids:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Unknown dependency: {dep}",
                        field="dependencies",
                        section_id=section.id,
                    ))

        # Check for circular dependencies
        if self._has_circular_deps(plan):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Circular dependencies detected",
                field="dependencies",
            ))

        return issues

    def _has_circular_deps(self, plan: StructuredPlan) -> bool:
        """Check for circular dependencies."""
        visited = set()
        rec_stack = set()

        def dfs(section_id: str) -> bool:
            visited.add(section_id)
            rec_stack.add(section_id)
            section = plan.get_section(section_id)
            if section:
                for dep in section.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            rec_stack.remove(section_id)
            return False

        for section in plan.sections:
            if section.id not in visited:
                if dfs(section.id):
                    return True
        return False


# ============================================================================
# Exports
# ============================================================================

# Aliases for backward compatibility
SectionPlan = Section
PlanConstraints = ContentConstraints


@dataclass
class PlanMetadata:
    """Metadata for a plan."""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    created_by: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "tags": self.tags,
        }


@dataclass
class EnrichmentConfig:
    """Configuration for content enrichment."""
    enable_research: bool = True
    enable_citations: bool = True
    enable_charts: bool = False
    enable_tables: bool = False
    enable_images: bool = False
    research_depth: str = "standard"
    max_sources_per_section: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_research": self.enable_research,
            "enable_citations": self.enable_citations,
            "research_depth": self.research_depth,
        }


@dataclass
class TemplateMatch:
    """Result of template matching."""
    template: DocumentTemplate
    score: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template.id,
            "template_name": self.template.name,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass
class TokenEstimate:
    """Token usage estimate."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ResourceEstimate:
    """Resource estimate for plan execution."""
    sections: int = 0
    research_queries: int = 0
    estimated_tokens: TokenEstimate = field(default_factory=TokenEstimate)
    estimated_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sections": self.sections,
            "research_queries": self.research_queries,
            "estimated_tokens": self.estimated_tokens.to_dict(),
            "estimated_time_seconds": self.estimated_time_seconds,
        }


__all__ = [
    # Enums
    "SectionType",
    "ContentType",
    "SourcePreference",
    "DocumentType",
    "FormalityLevel",
    "WritingStyle",
    "Tone",
    "ValidationSeverity",
    # Data classes
    "StyleGuide",
    "ResearchConstraints",
    "ContentConstraints",
    "Microprompt",
    "Section",
    "StructuredPlan",
    "TemplateSection",
    "DocumentTemplate",
    "ValidationIssue",
    "ValidationResult",
    # Additional classes
    "SectionPlan",
    "PlanConstraints",
    "PlanMetadata",
    "EnrichmentConfig",
    "TemplateMatch",
    "TokenEstimate",
    "PlanTokenEstimate",
    "PlanResourceEstimate",
    "ResourceEstimate",
    # Managers
    "TemplateManager",
    "MicropromptEngine",
    "PlanGenerator",
    "PlanValidator",
]
