"""
citation_manager/providers/__init__.py

Data classes and enums for citation management.

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


class CitationStyle(str, Enum):
    """Citation formatting styles."""
    APA = "apa"
    APA7 = "apa7"
    MLA = "mla"
    MLA9 = "mla9"
    CHICAGO = "chicago"
    CHICAGO_NOTES = "chicago_notes"
    HARVARD = "harvard"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    AMA = "ama"
    NUMERIC = "numeric"
    AUTHOR_DATE = "author_date"
    FOOTNOTE = "footnote"


class SourceType(str, Enum):
    """Type of citation source."""
    BOOK = "book"
    JOURNAL = "journal"
    ARTICLE = "article"
    CONFERENCE = "conference"
    WEBSITE = "website"
    REPORT = "report"
    THESIS = "thesis"
    PATENT = "patent"
    LEGISLATION = "legislation"
    DATABASE = "database"
    SOFTWARE = "software"
    INTERVIEW = "interview"
    RAG_DOCUMENT = "rag_document"
    OTHER = "other"


class CitationStatus(str, Enum):
    """Status of a citation."""
    ACTIVE = "active"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DUPLICATE = "duplicate"
    INVALID = "invalid"


# ============================================================================
# Citation Data Classes
# ============================================================================


@dataclass
class Author:
    """Author information."""
    last_name: str
    first_name: str = ""
    middle_name: str = ""
    suffix: str = ""  # Jr., III, etc.
    orcid: Optional[str] = None
    affiliation: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(p for p in parts if p)
    
    @property
    def citation_name(self) -> str:
        """Name formatted for citations (Last, F. M.)"""
        initials = ""
        if self.first_name:
            initials += f"{self.first_name[0]}."
        if self.middle_name:
            initials += f" {self.middle_name[0]}."
        return f"{self.last_name}, {initials}".strip()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_name": self.last_name,
            "first_name": self.first_name,
            "full_name": self.full_name,
        }


@dataclass
class Citation:
    """
    Complete citation record.
    
    Supports various source types with comprehensive metadata.
    """
    # Identity
    id: str
    
    # Core fields
    title: str
    authors: List[Author] = field(default_factory=list)
    year: Optional[str] = None
    
    # Source type
    source_type: SourceType = SourceType.OTHER
    
    # Publication info
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None
    
    # Digital identifiers
    doi: Optional[str] = None
    url: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None
    pmid: Optional[str] = None  # PubMed ID
    arxiv: Optional[str] = None
    
    # Access info
    accessed_date: Optional[str] = None
    
    # Location
    city: Optional[str] = None
    country: Optional[str] = None
    
    # RAG-specific
    collection: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    relevance_score: float = 0.0
    
    # Content excerpt
    excerpt: str = ""
    context: str = ""
    
    # Status
    status: CitationStatus = CitationStatus.ACTIVE
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Internal tracking
    usage_count: int = 0
    section_ids: List[str] = field(default_factory=list)
    
    def get_authors_string(self, style: CitationStyle = CitationStyle.APA) -> str:
        """Format authors for citation style."""
        if not self.authors:
            return "Unknown"
        
        if style in [CitationStyle.APA, CitationStyle.APA7]:
            if len(self.authors) == 1:
                return self.authors[0].citation_name
            elif len(self.authors) == 2:
                return f"{self.authors[0].citation_name} & {self.authors[1].citation_name}"
            else:
                return f"{self.authors[0].citation_name} et al."
        
        elif style == CitationStyle.IEEE:
            names = [f"{a.first_name[0]}. {a.last_name}" if a.first_name else a.last_name 
                     for a in self.authors[:3]]
            if len(self.authors) > 3:
                names.append("et al.")
            return ", ".join(names)
        
        else:
            return ", ".join(a.full_name for a in self.authors)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "year": self.year,
            "source_type": self.source_type.value,
            "journal": self.journal,
            "doi": self.doi,
            "url": self.url,
            "collection": self.collection,
            "relevance_score": self.relevance_score,
            "status": self.status.value,
            "usage_count": self.usage_count,
        }


@dataclass
class InlineCitation:
    """An inline citation reference."""
    citation_id: str
    number: int  # For numeric styles
    position: int = 0  # Position in text
    page: Optional[str] = None  # Specific page reference
    
    def format(self, style: CitationStyle) -> str:
        if style == CitationStyle.NUMERIC:
            return f"[{self.number}]"
        elif style == CitationStyle.FOOTNOTE:
            return f"^{self.number}"
        else:
            return f"[{self.number}]"


@dataclass
class Bibliography:
    """Complete bibliography."""
    citations: List[Citation] = field(default_factory=list)
    style: CitationStyle = CitationStyle.APA
    title: str = "Bibliografia"
    
    # Formatting options
    numbered: bool = True
    sorted_by: str = "author"  # author, year, order
    group_by_type: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style.value,
            "title": self.title,
            "count": len(self.citations),
            "numbered": self.numbered,
        }


# ============================================================================
# Validation
# ============================================================================


@dataclass
class ValidationIssue:
    """A validation issue with a citation."""
    citation_id: str
    field: str
    severity: str  # error, warning, info
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of citation validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "issues": [
                {
                    "citation_id": i.citation_id,
                    "field": i.field,
                    "severity": i.severity,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


# ============================================================================
# Export/Import
# ============================================================================


@dataclass
class ExportResult:
    """Result of export operation."""
    success: bool
    format: str
    content: str = ""
    error: Optional[str] = None


@dataclass
class ImportResult:
    """Result of import operation."""
    success: bool
    imported_count: int = 0
    skipped_count: int = 0
    errors: List[str] = field(default_factory=list)


# Export all
__all__ = [
    # Enums
    "CitationStyle",
    "SourceType",
    "CitationStatus",
    # Data classes
    "Author",
    "Citation",
    "InlineCitation",
    "Bibliography",
    # Validation
    "ValidationIssue",
    "ValidationResult",
    # Export/Import
    "ExportResult",
    "ImportResult",
]
