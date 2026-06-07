"""
content_planner/providers/__init__.py

Data classes and enums for content planning.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


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
    PROSE = "prose"              # Narrative text
    TABLE = "table"              # Tabular data
    CHART = "chart"              # Visualization
    LIST = "list"                # Bullet/numbered list
    MIXED = "mixed"              # Combination
    DATA_DRIVEN = "data_driven"  # Heavy on data/numbers


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
    INFORMATIVE = "informative"


# ============================================================================
# Microprompt Data Classes
# ============================================================================


@dataclass
class Microprompt:
    """
    Specific prompt for generating section content.
    
    Contains all guidance needed for LLM to generate
    high-quality, consistent section content.
    """
    section_id: str
    section_type: SectionType
    
    # Core prompts
    system_context: str
    generation_prompt: str
    
    # Style guidance
    writing_style: WritingStyle = WritingStyle.FORMAL
    tone: Tone = Tone.NEUTRAL
    
    # Structure hints
    structure_elements: List[str] = field(default_factory=list)
    required_elements: List[str] = field(default_factory=list)
    optional_elements: List[str] = field(default_factory=list)
    avoid_elements: List[str] = field(default_factory=list)
    
    # Content guidance
    include_citations: bool = True
    include_examples: bool = False
    include_data: bool = False
    include_recommendations: bool = False
    
    # Length guidance
    min_tokens: int = 100
    max_tokens: int = 1000
    target_paragraphs: Optional[int] = None
    
    # Custom instructions
    custom_instructions: Optional[str] = None
    
    # Quality criteria
    quality_criteria: List[str] = field(default_factory=list)
    
    def to_system_prompt(self) -> str:
        """Build complete system prompt for LLM."""
        parts = [self.system_context]
        
        if self.writing_style:
            parts.append(f"\nWriting Style: {self.writing_style.value}")
        if self.tone:
            parts.append(f"Tone: {self.tone.value}")
        
        if self.structure_elements:
            parts.append(f"\nStructure: Include these elements: {', '.join(self.structure_elements)}")
        
        if self.required_elements:
            parts.append(f"Required: {', '.join(self.required_elements)}")
        
        if self.avoid_elements:
            parts.append(f"Avoid: {', '.join(self.avoid_elements)}")
        
        if self.custom_instructions:
            parts.append(f"\nAdditional Instructions: {self.custom_instructions}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_type": self.section_type.value,
            "system_context": self.system_context,
            "generation_prompt": self.generation_prompt,
            "writing_style": self.writing_style.value,
            "tone": self.tone.value,
            "structure_elements": self.structure_elements,
            "required_elements": self.required_elements,
            "avoid_elements": self.avoid_elements,
            "include_citations": self.include_citations,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "custom_instructions": self.custom_instructions,
        }


# ============================================================================
# Section Data Classes
# ============================================================================


@dataclass
class EnrichmentConfig:
    """Enrichment configuration for section research."""
    enable_hyde: bool = False
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    custom_top_k: Optional[int] = None
    custom_min_score: Optional[float] = None
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class SectionPlan:
    """
    Plan for a single document section.
    
    Contains all information needed to research and
    generate content for this section.
    """
    id: str
    title: str
    description: str
    order: int
    
    # Type and format
    section_type: SectionType = SectionType.CUSTOM
    content_type: ContentType = ContentType.PROSE
    
    # Content guidance
    microprompt: Optional[Microprompt] = None
    
    # Research configuration
    source_preference: SourcePreference = SourcePreference.RAG_FIRST
    suggested_queries: List[str] = field(default_factory=list)
    required_data_types: List[str] = field(default_factory=list)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    provides_context_for: List[str] = field(default_factory=list)
    
    # Token budget
    min_tokens: int = 100
    max_tokens: int = 1000
    target_tokens: int = 500
    
    # Customization
    custom_template: Optional[str] = None
    style_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Enrichment
    enrichment_config: Optional[EnrichmentConfig] = None
    
    # Flags
    required: bool = True
    interactive_review: bool = False
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "order": self.order,
            "section_type": self.section_type.value,
            "content_type": self.content_type.value,
            "microprompt": self.microprompt.to_dict() if self.microprompt else None,
            "source_preference": self.source_preference.value,
            "suggested_queries": self.suggested_queries,
            "required_data_types": self.required_data_types,
            "depends_on": self.depends_on,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "target_tokens": self.target_tokens,
            "required": self.required,
            "interactive_review": self.interactive_review,
            "enabled": self.enabled,
        }


# ============================================================================
# Plan Data Classes
# ============================================================================


@dataclass
class PlanConstraints:
    """Constraints for document planning."""
    max_sections: int = 15
    max_tokens_total: int = 15000
    max_tokens_per_section: int = 3000
    min_sections: int = 2
    required_section_types: List[SectionType] = field(default_factory=list)
    forbidden_topics: List[str] = field(default_factory=list)
    target_audience: str = "general"
    formality_level: FormalityLevel = FormalityLevel.PROFESSIONAL
    language: str = "it"  # Default Italian
    include_citations: bool = True
    include_toc: bool = True


@dataclass
class PlanMetadata:
    """Metadata for the plan."""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    generation_model: Optional[str] = None
    planning_time_ms: Optional[int] = None
    version: int = 1


@dataclass
class StructuredPlan:
    """
    Complete structured plan for a document.
    
    This is the main output of the content_planner module,
    containing all information needed to research and generate
    the complete document.
    """
    id: str
    title: str
    description: str
    
    # Document type
    document_type: DocumentType = DocumentType.REPORT
    
    # Sections
    sections: List[SectionPlan] = field(default_factory=list)
    
    # Configuration
    constraints: PlanConstraints = field(default_factory=PlanConstraints)
    metadata: PlanMetadata = field(default_factory=PlanMetadata)
    
    # Estimates
    estimated_tokens: int = 0
    estimated_sections: int = 0
    estimated_time_minutes: int = 0
    
    # Collections for research
    collections: List[str] = field(default_factory=list)
    
    # Language
    language: str = "it"
    
    @property
    def enabled_sections(self) -> List[SectionPlan]:
        """Get only enabled sections."""
        return [s for s in self.sections if s.enabled]
    
    @property
    def required_sections(self) -> List[SectionPlan]:
        """Get required sections."""
        return [s for s in self.sections if s.required and s.enabled]
    
    def get_section(self, section_id: str) -> Optional[SectionPlan]:
        """Get section by ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None
    
    def get_sections_by_type(self, section_type: SectionType) -> List[SectionPlan]:
        """Get sections by type."""
        return [s for s in self.sections if s.section_type == section_type]
    
    def get_dependency_order(self) -> List[List[SectionPlan]]:
        """
        Get sections grouped by dependency level for parallel execution.
        Returns list of batches that can be executed in parallel.
        """
        # Build dependency graph
        remaining = {s.id: s for s in self.enabled_sections}
        completed = set()
        batches = []
        
        while remaining:
            # Find sections with all dependencies satisfied
            batch = []
            for section_id, section in list(remaining.items()):
                deps = set(section.depends_on)
                if deps.issubset(completed):
                    batch.append(section)
            
            if not batch:
                # Circular dependency or missing dependency
                # Add remaining in order
                batch = sorted(remaining.values(), key=lambda s: s.order)
                batches.append(batch)
                break
            
            # Sort batch by order
            batch.sort(key=lambda s: s.order)
            batches.append(batch)
            
            # Mark as completed
            for section in batch:
                completed.add(section.id)
                del remaining[section.id]
        
        return batches
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "document_type": self.document_type.value,
            "sections": [s.to_dict() for s in self.sections],
            "constraints": {
                "max_sections": self.constraints.max_sections,
                "max_tokens_total": self.constraints.max_tokens_total,
                "formality_level": self.constraints.formality_level.value,
                "language": self.constraints.language,
            },
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "template_id": self.metadata.template_id,
                "version": self.metadata.version,
            },
            "estimated_tokens": self.estimated_tokens,
            "estimated_sections": len(self.enabled_sections),
            "collections": self.collections,
            "language": self.language,
        }


# ============================================================================
# Template Data Classes
# ============================================================================


@dataclass
class SectionTemplate:
    """Template for a section within a document template."""
    id: str
    title_template: str  # Can contain {topic}, {subject} placeholders
    section_type: SectionType
    content_type: ContentType
    
    # Microprompt template
    microprompt_template: str
    writing_style: WritingStyle = WritingStyle.FORMAL
    tone: Tone = Tone.NEUTRAL
    
    # Structure
    structure_elements: List[str] = field(default_factory=list)
    required_elements: List[str] = field(default_factory=list)
    
    # Research
    source_preference: SourcePreference = SourcePreference.RAG_FIRST
    query_templates: List[str] = field(default_factory=list)
    
    # Tokens
    default_tokens: int = 500
    min_tokens: int = 100
    max_tokens: int = 1500
    
    # Flags
    required: bool = True
    order: int = 0


@dataclass
class DocumentTemplate:
    """
    Complete template for a document type.
    
    Templates define the structure, style, and content
    guidance for specific document types.
    """
    id: str
    name: str
    description: str
    category: str
    document_type: DocumentType
    
    # Sections
    sections: List[SectionTemplate] = field(default_factory=list)
    
    # Selection criteria
    keywords: List[str] = field(default_factory=list)
    selection_patterns: List[str] = field(default_factory=list)
    domain_hints: List[str] = field(default_factory=list)
    
    # Style
    default_formality: FormalityLevel = FormalityLevel.PROFESSIONAL
    default_language: str = "it"
    
    # Document styling
    docx_template_path: Optional[str] = None
    pptx_template_path: Optional[str] = None
    style_config: Dict[str, Any] = field(default_factory=dict)
    
    # Constraints
    max_sections: int = 15
    default_max_tokens: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "document_type": self.document_type.value,
            "keywords": self.keywords,
            "sections_count": len(self.sections),
            "default_formality": self.default_formality.value,
        }


# ============================================================================
# Validation Data Classes
# ============================================================================


@dataclass
class ValidationIssue:
    """A validation issue found in the plan."""
    severity: str  # error, warning, info
    code: str
    message: str
    section_id: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of plan validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "section_id": i.section_id,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "warnings_count": self.warnings_count,
            "errors_count": self.errors_count,
        }


# ============================================================================
# Estimation Data Classes
# ============================================================================


@dataclass
class TokenEstimate:
    """Token estimation for plan."""
    total_tokens: int
    tokens_by_section: Dict[str, int]
    estimated_cost_usd: Optional[float] = None
    within_budget: bool = True
    budget_utilization: float = 0.0  # 0-1


@dataclass
class ResourceEstimate:
    """Resource estimation for plan execution."""
    estimated_time_minutes: int
    estimated_api_calls: int
    estimated_tokens: TokenEstimate
    parallel_batches: int
    sections_requiring_research: int
    sections_llm_only: int


# ============================================================================
# Match Result
# ============================================================================


@dataclass
class TemplateMatch:
    """Result of template matching."""
    template_id: str
    template_name: str
    confidence: float  # 0-1
    match_reasons: List[str]
    matched_keywords: List[str]
    matched_patterns: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "confidence": self.confidence,
            "match_reasons": self.match_reasons,
        }


# Export all
__all__ = [
    # Enums
    "SectionType",
    "ContentType",
    "SourcePreference",
    "DocumentType",
    "FormalityLevel",
    "WritingStyle",
    "Tone",
    # Data classes
    "Microprompt",
    "EnrichmentConfig",
    "SectionPlan",
    "PlanConstraints",
    "PlanMetadata",
    "StructuredPlan",
    "SectionTemplate",
    "DocumentTemplate",
    "ValidationIssue",
    "ValidationResult",
    "TokenEstimate",
    "ResourceEstimate",
    "TemplateMatch",
]
