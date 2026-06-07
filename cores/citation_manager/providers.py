"""
citation_manager/providers.py

Domain Layer - Data classes and logic for citation management.

Contains:
- Enums: CitationStyle, SourceType, CitationStatus
- Data classes: Author, Citation, InlineCitation, Bibliography, ValidationIssue, ValidationResult
- Components: CitationStore, CitationFormatter, CitationValidator, DuplicateDetector

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class CitationStyle(str, Enum):
    """Supported citation styles."""
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
    """Types of citation sources."""
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


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================================
# Core Data Classes
# ============================================================================


@dataclass
class Author:
    """Author information."""
    last_name: str
    first_name: str = ""
    middle_name: str = ""
    suffix: str = ""
    orcid: Optional[str] = None
    affiliation: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get full name."""
        parts = [self.first_name, self.middle_name, self.last_name]
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(p for p in parts if p)

    @property
    def citation_name(self) -> str:
        """Get name for citation (Last, F. M.)."""
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
    """A citation record."""
    id: str
    title: str
    authors: List[Author] = field(default_factory=list)
    year: Optional[str] = None
    source_type: SourceType = SourceType.OTHER
    
    # Publication info
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None
    
    # Identifiers
    doi: Optional[str] = None
    url: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None
    pmid: Optional[str] = None
    arxiv: Optional[str] = None
    
    # Additional
    accessed_date: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    
    # RAG specific
    collection: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    relevance_score: float = 0.0
    excerpt: str = ""
    context: str = ""
    
    # Metadata
    status: CitationStatus = CitationStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    section_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "year": self.year,
            "source_type": self.source_type.value,
            "doi": self.doi,
            "url": self.url,
            "status": self.status.value,
        }


@dataclass
class InlineCitation:
    """Inline citation reference."""
    citation_id: str
    number: Optional[int] = None
    position: int = 0
    page: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "number": self.number,
            "position": self.position,
        }


@dataclass
class Bibliography:
    """Formatted bibliography."""
    citations: List[Citation] = field(default_factory=list)
    style: CitationStyle = CitationStyle.APA
    title: str = "References"
    numbered: bool = True
    sorted_by: str = "author"
    group_by_type: bool = False
    formatted_entries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_count": len(self.citations),
            "style": self.style.value,
            "title": self.title,
            "numbered": self.numbered,
        }


# ============================================================================
# Validation Classes
# ============================================================================


@dataclass
class ValidationIssue:
    """A validation issue."""
    citation_id: str
    field: str
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "field": self.field,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class ExportResult:
    """Result of export operation."""
    success: bool
    format: str
    content: str = ""
    path: Optional[str] = None
    citation_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "format": self.format,
            "citation_count": self.citation_count,
            "error": self.error,
        }


@dataclass
class ImportResult:
    """Result of import operation."""
    success: bool
    imported_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    citations: List[Citation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "imported_count": self.imported_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
        }


# ============================================================================
# Citation Store
# ============================================================================


class CitationStore:
    """Stores and manages citations."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path
        self._citations: Dict[str, Citation] = {}
        self._by_doi: Dict[str, str] = {}
        self._by_url: Dict[str, str] = {}
        self._by_title: Dict[str, str] = {}
        self._by_document: Dict[str, List[str]] = {}
        self._by_section: Dict[str, List[str]] = {}
        self._by_tag: Dict[str, List[str]] = {}
        
        if persist_path and persist_path.exists():
            self._load()

    def add(self, citation: Citation, check_duplicates: bool = True) -> Tuple[bool, Optional[str]]:
        """Add citation. Returns (success, duplicate_id if found)."""
        if check_duplicates:
            dup_id = self._find_duplicate(citation)
            if dup_id:
                return False, dup_id
        
        self._citations[citation.id] = citation
        self._index_citation(citation)
        self._save()
        return True, None

    def get(self, citation_id: str) -> Optional[Citation]:
        """Get citation by ID."""
        return self._citations.get(citation_id)

    def update(self, citation: Citation) -> bool:
        """Update existing citation."""
        if citation.id not in self._citations:
            return False
        
        self._unindex_citation(self._citations[citation.id])
        self._citations[citation.id] = citation
        self._index_citation(citation)
        self._save()
        return True

    def delete(self, citation_id: str) -> bool:
        """Delete citation."""
        if citation_id not in self._citations:
            return False
        
        self._unindex_citation(self._citations[citation_id])
        del self._citations[citation_id]
        self._save()
        return True

    def list_all(self) -> List[Citation]:
        """List all citations."""
        return list(self._citations.values())

    def search(
        self,
        query: Optional[str] = None,
        source_type: Optional[SourceType] = None,
        status: Optional[CitationStatus] = None,
        tags: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        section_id: Optional[str] = None,
        min_relevance: Optional[float] = None,
        limit: int = 100,
    ) -> List[Citation]:
        """Search citations with filters."""
        results = list(self._citations.values())
        
        if source_type:
            results = [c for c in results if c.source_type == source_type]
        
        if status:
            results = [c for c in results if c.status == status]
        
        if tags:
            results = [c for c in results if any(t in c.tags for t in tags)]
        
        if document_id:
            ids = self._by_document.get(document_id, [])
            results = [c for c in results if c.id in ids]
        
        if section_id:
            ids = self._by_section.get(section_id, [])
            results = [c for c in results if c.id in ids]
        
        if min_relevance is not None:
            results = [c for c in results if c.relevance_score >= min_relevance]
        
        if query:
            query_lower = query.lower()
            results = [c for c in results if 
                       query_lower in c.title.lower() or
                       any(query_lower in a.full_name.lower() for a in c.authors)]
        
        return results[:limit]

    def _find_duplicate(self, citation: Citation) -> Optional[str]:
        """Find duplicate citation."""
        # Check DOI
        if citation.doi:
            if citation.doi.lower() in self._by_doi:
                return self._by_doi[citation.doi.lower()]
        
        # Check URL
        if citation.url:
            normalized = self._normalize_url(citation.url)
            if normalized in self._by_url:
                return self._by_url[normalized]
        
        # Check title similarity
        normalized_title = self._normalize_title(citation.title)
        if normalized_title in self._by_title:
            return self._by_title[normalized_title]
        
        return None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        url = url.lower().strip()
        url = re.sub(r'^https?://', '', url)
        url = url.rstrip('/')
        return url

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title[:100]

    def _index_citation(self, citation: Citation) -> None:
        """Index citation for fast lookup."""
        if citation.doi:
            self._by_doi[citation.doi.lower()] = citation.id
        if citation.url:
            self._by_url[self._normalize_url(citation.url)] = citation.id
        self._by_title[self._normalize_title(citation.title)] = citation.id
        
        if citation.document_id:
            if citation.document_id not in self._by_document:
                self._by_document[citation.document_id] = []
            self._by_document[citation.document_id].append(citation.id)
        
        for section_id in citation.section_ids:
            if section_id not in self._by_section:
                self._by_section[section_id] = []
            self._by_section[section_id].append(citation.id)
        
        for tag in citation.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(citation.id)

    def _unindex_citation(self, citation: Citation) -> None:
        """Remove citation from indexes."""
        if citation.doi and citation.doi.lower() in self._by_doi:
            del self._by_doi[citation.doi.lower()]
        if citation.url:
            normalized = self._normalize_url(citation.url)
            if normalized in self._by_url:
                del self._by_url[normalized]
        normalized_title = self._normalize_title(citation.title)
        if normalized_title in self._by_title:
            del self._by_title[normalized_title]

    def _save(self) -> None:
        """Save to disk."""
        if not self.persist_path:
            return
        data = {
            cid: {
                **c.to_dict(),
                "created_at": c.created_at.isoformat(),
            }
            for cid, c in self._citations.items()
        }
        self.persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
            for cid, cdata in data.items():
                citation = self._dict_to_citation(cdata)
                self._citations[cid] = citation
                self._index_citation(citation)
        except Exception as e:
            logger.error(f"Failed to load citations: {e}")

    def _dict_to_citation(self, data: Dict[str, Any]) -> Citation:
        """Convert dict to Citation."""
        authors = [
            Author(
                last_name=a.get("last_name", ""),
                first_name=a.get("first_name", ""),
            )
            for a in data.get("authors", [])
        ]
        return Citation(
            id=data.get("id", ""),
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year"),
            source_type=SourceType(data.get("source_type", "other")),
            doi=data.get("doi"),
            url=data.get("url"),
            status=CitationStatus(data.get("status", "active")),
        )


# ============================================================================
# Citation Formatter
# ============================================================================


class CitationFormatter:
    """Formats citations in various styles."""

    def format(self, citation: Citation, style: CitationStyle) -> str:
        """Format a citation in the specified style."""
        formatter = getattr(self, f"_format_{style.value}", self._format_apa)
        return formatter(citation)

    def format_inline(
        self,
        citation: Citation,
        style: CitationStyle,
        number: Optional[int] = None,
    ) -> str:
        """Format inline citation."""
        if style in (CitationStyle.NUMERIC, CitationStyle.IEEE, CitationStyle.VANCOUVER):
            return f"[{number}]" if number else "[?]"
        elif style in (CitationStyle.APA, CitationStyle.APA7, CitationStyle.HARVARD):
            author = citation.authors[0].last_name if citation.authors else "Unknown"
            year = citation.year or "n.d."
            return f"({author}, {year})"
        else:
            author = citation.authors[0].last_name if citation.authors else "Unknown"
            return f"({author})"

    def _format_apa(self, citation: Citation) -> str:
        """Format in APA style."""
        parts = []
        
        # Authors
        if citation.authors:
            if len(citation.authors) == 1:
                parts.append(citation.authors[0].citation_name)
            elif len(citation.authors) == 2:
                parts.append(f"{citation.authors[0].citation_name} & {citation.authors[1].citation_name}")
            else:
                parts.append(f"{citation.authors[0].citation_name} et al.")
        
        # Year
        if citation.year:
            parts.append(f"({citation.year}).")
        else:
            parts.append("(n.d.).")
        
        # Title
        if citation.source_type == SourceType.JOURNAL:
            parts.append(citation.title + ".")
        else:
            parts.append(f"*{citation.title}*.")
        
        # Journal/Publisher
        if citation.journal:
            parts.append(f"*{citation.journal}*")
            if citation.volume:
                parts.append(f", {citation.volume}")
            if citation.issue:
                parts.append(f"({citation.issue})")
            if citation.pages:
                parts.append(f", {citation.pages}")
            parts.append(".")
        elif citation.publisher:
            parts.append(f"{citation.publisher}.")
        
        # DOI/URL
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(citation.url)
        
        return " ".join(parts)

    def _format_mla(self, citation: Citation) -> str:
        """Format in MLA style."""
        parts = []
        
        # Authors
        if citation.authors:
            parts.append(f"{citation.authors[0].last_name}, {citation.authors[0].first_name}.")
        
        # Title
        parts.append(f'"{citation.title}."')
        
        # Container (journal)
        if citation.journal:
            parts.append(f"*{citation.journal}*,")
            if citation.volume:
                parts.append(f"vol. {citation.volume},")
            if citation.issue:
                parts.append(f"no. {citation.issue},")
        
        # Year
        if citation.year:
            parts.append(f"{citation.year},")
        
        # Pages
        if citation.pages:
            parts.append(f"pp. {citation.pages}.")
        
        return " ".join(parts)

    def _format_ieee(self, citation: Citation) -> str:
        """Format in IEEE style."""
        parts = []
        
        # Authors
        if citation.authors:
            author_strs = []
            for a in citation.authors[:3]:
                first_init = f"{a.first_name[0]}." if a.first_name else ""
                author_strs.append(f"{first_init} {a.last_name}")
            if len(citation.authors) > 3:
                author_strs.append("et al.")
            parts.append(", ".join(author_strs) + ",")
        
        # Title
        parts.append(f'"{citation.title},"')
        
        # Journal
        if citation.journal:
            parts.append(f"*{citation.journal}*,")
        
        # Volume, pages, year
        vol_parts = []
        if citation.volume:
            vol_parts.append(f"vol. {citation.volume}")
        if citation.issue:
            vol_parts.append(f"no. {citation.issue}")
        if citation.pages:
            vol_parts.append(f"pp. {citation.pages}")
        if citation.year:
            vol_parts.append(citation.year)
        if vol_parts:
            parts.append(", ".join(vol_parts) + ".")
        
        return " ".join(parts)

    def _format_harvard(self, citation: Citation) -> str:
        """Format in Harvard style."""
        return self._format_apa(citation)  # Similar to APA

    def _format_chicago(self, citation: Citation) -> str:
        """Format in Chicago style."""
        parts = []
        
        if citation.authors:
            parts.append(f"{citation.authors[0].last_name}, {citation.authors[0].first_name}.")
        
        parts.append(f"*{citation.title}*.")
        
        if citation.city and citation.publisher:
            parts.append(f"{citation.city}: {citation.publisher},")
        elif citation.publisher:
            parts.append(f"{citation.publisher},")
        
        if citation.year:
            parts.append(f"{citation.year}.")
        
        return " ".join(parts)

    def _format_numeric(self, citation: Citation) -> str:
        """Format for numeric style."""
        return self._format_apa(citation)

    def _format_author_date(self, citation: Citation) -> str:
        """Format for author-date style."""
        return self._format_apa(citation)


# ============================================================================
# BibTeX Formatter
# ============================================================================


class BibTeXFormatter:
    """Formats citations as BibTeX."""

    TYPE_MAP = {
        SourceType.BOOK: "book",
        SourceType.JOURNAL: "article",
        SourceType.ARTICLE: "article",
        SourceType.CONFERENCE: "inproceedings",
        SourceType.THESIS: "phdthesis",
        SourceType.REPORT: "techreport",
        SourceType.WEBSITE: "misc",
    }

    def format(self, citation: Citation) -> str:
        """Format citation as BibTeX entry."""
        bib_type = self.TYPE_MAP.get(citation.source_type, "misc")
        key = self._generate_key(citation)
        
        lines = [f"@{bib_type}{{{key},"]
        
        # Authors
        if citation.authors:
            author_str = " and ".join(a.full_name for a in citation.authors)
            lines.append(f"  author = {{{author_str}}},")
        
        # Title
        lines.append(f"  title = {{{citation.title}}},")
        
        # Year
        if citation.year:
            lines.append(f"  year = {{{citation.year}}},")
        
        # Journal
        if citation.journal:
            lines.append(f"  journal = {{{citation.journal}}},")
        
        # Volume, Issue, Pages
        if citation.volume:
            lines.append(f"  volume = {{{citation.volume}}},")
        if citation.issue:
            lines.append(f"  number = {{{citation.issue}}},")
        if citation.pages:
            lines.append(f"  pages = {{{citation.pages}}},")
        
        # Publisher
        if citation.publisher:
            lines.append(f"  publisher = {{{citation.publisher}}},")
        
        # DOI/URL
        if citation.doi:
            lines.append(f"  doi = {{{citation.doi}}},")
        if citation.url:
            lines.append(f"  url = {{{citation.url}}},")
        
        lines.append("}")
        return "\n".join(lines)

    def _generate_key(self, citation: Citation) -> str:
        """Generate BibTeX key."""
        author = citation.authors[0].last_name.lower() if citation.authors else "unknown"
        year = citation.year or "nd"
        author = re.sub(r'[^a-z]', '', author)
        return f"{author}{year}"


# ============================================================================
# Citation Validator
# ============================================================================


class CitationValidator:
    """Validates citations."""

    def validate(self, citation: Citation) -> ValidationResult:
        """Validate a citation."""
        issues = []
        
        # Required: title
        if not citation.title:
            issues.append(ValidationIssue(
                citation_id=citation.id,
                field="title",
                severity=ValidationSeverity.ERROR,
                message="Title is required",
            ))
        
        # Recommended: authors
        if not citation.authors:
            if citation.source_type not in (SourceType.WEBSITE, SourceType.DATABASE):
                issues.append(ValidationIssue(
                    citation_id=citation.id,
                    field="authors",
                    severity=ValidationSeverity.WARNING,
                    message="Authors recommended",
                    suggestion="Add at least one author",
                ))
        
        # Recommended: year
        if not citation.year:
            issues.append(ValidationIssue(
                citation_id=citation.id,
                field="year",
                severity=ValidationSeverity.WARNING,
                message="Year recommended",
            ))
        elif not re.match(r'^\d{4}$', str(citation.year)):  # Bug 3 Fix: Convert to string before regex
            issues.append(ValidationIssue(
                citation_id=citation.id,
                field="year",
                severity=ValidationSeverity.WARNING,
                message="Year should be 4 digits",
            ))
        
        # Format validation: DOI
        if citation.doi and not re.match(r'^10\.\d{4,}/', citation.doi):
            issues.append(ValidationIssue(
                citation_id=citation.id,
                field="doi",
                severity=ValidationSeverity.WARNING,
                message="DOI format may be incorrect",
                suggestion="DOI should start with 10.xxxx/",
            ))
        
        # Format validation: URL
        if citation.url and not citation.url.startswith(('http://', 'https://')):
            issues.append(ValidationIssue(
                citation_id=citation.id,
                field="url",
                severity=ValidationSeverity.WARNING,
                message="URL should include protocol",
                suggestion="Add http:// or https://",
            ))
        
        # Source-specific validation
        if citation.source_type == SourceType.JOURNAL and not citation.journal:
            issues.append(ValidationIssue(
                citation_id=citation.id,
                field="journal",
                severity=ValidationSeverity.WARNING,
                message="Journal name recommended for journal articles",
            ))
        
        if citation.source_type == SourceType.WEBSITE and not citation.accessed_date:
            issues.append(ValidationIssue(
                citation_id=citation.id,
                field="accessed_date",
                severity=ValidationSeverity.INFO,
                message="Access date recommended for websites",
            ))
        
        return ValidationResult(
            valid=len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0,
            issues=issues,
        )

    def validate_all(self, citations: List[Citation]) -> Dict[str, ValidationResult]:
        """Validate multiple citations."""
        return {c.id: self.validate(c) for c in citations}


# ============================================================================
# Duplicate Detector
# ============================================================================


class DuplicateDetector:
    """Detects duplicate citations."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    def find_duplicates(self, citations: List[Citation]) -> List[Tuple[str, str, str]]:
        """Find duplicate pairs. Returns list of (id1, id2, reason)."""
        duplicates = []
        
        for i, c1 in enumerate(citations):
            for c2 in citations[i + 1:]:
                reason = self._check_duplicate(c1, c2)
                if reason:
                    duplicates.append((c1.id, c2.id, reason))
        
        return duplicates

    def _check_duplicate(self, c1: Citation, c2: Citation) -> Optional[str]:
        """Check if two citations are duplicates."""
        # Exact DOI match
        if c1.doi and c2.doi and c1.doi.lower() == c2.doi.lower():
            return "same_doi"
        
        # Exact URL match
        if c1.url and c2.url:
            if self._normalize_url(c1.url) == self._normalize_url(c2.url):
                return "same_url"
        
        # Title similarity
        sim = self._title_similarity(c1.title, c2.title)
        if sim >= self.threshold:
            # Also check first author
            if c1.authors and c2.authors:
                if c1.authors[0].last_name.lower() == c2.authors[0].last_name.lower():
                    return f"similar_title ({sim:.2f})"
        
        return None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL."""
        url = url.lower().strip()
        url = re.sub(r'^https?://', '', url)
        return url.rstrip('/')

    def _title_similarity(self, t1: str, t2: str) -> float:
        """Calculate Jaccard similarity of titles."""
        words1 = set(re.findall(r'\w+', t1.lower()))
        words2 = set(re.findall(r'\w+', t2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Enums
    "CitationStyle",
    "SourceType",
    "CitationStatus",
    "ValidationSeverity",
    # Data classes
    "Author",
    "Citation",
    "InlineCitation",
    "Bibliography",
    "ValidationIssue",
    "ValidationResult",
    "ExportResult",
    "ImportResult",
    # Components
    "CitationStore",
    "SessionScopedCitationStore",
    "CitationFormatter",
    "BibTeXFormatter",
    "CitationValidator",
    "DuplicateDetector",
]


# ---------------------------------------------------------------------------
# fb2 (B-07) — Session-scoped Redis-backed citation store
# ---------------------------------------------------------------------------
# Rationale: the in-memory CitationStore above is process-global, which means
# citations from session A can leak into the synthesis context of session B
# running on the same backend worker. The audit (B-07) ties this to the chat
# 2026-05-06 incident where pack_research surfaced URLs unrelated to the
# active session.
#
# This wrapper stores citations in a Redis hash keyed on session_id with a
# 24h TTL. Storage format mirrors Citation.to_dict() so existing formatters
# (CitationFormatter, BibTeXFormatter) keep working unmodified once
# integrated. fc3 (Fase C) will wire this store into the pack_research
# synthesis flow; fb2 only ships the storage primitive.

class SessionScopedCitationStore:
    """Redis-backed, session-scoped citation store with 24h TTL.

    Each session_id maps to a Redis hash where the field is the citation
    id and the value is the JSON-serialised Citation.to_dict() payload.

    The class is intentionally synchronous to mirror the existing
    CitationStore API surface; the redis client passed in the constructor
    must therefore be a sync redis.Redis instance (or any object exposing
    the subset hset/hget/hgetall/hdel/expire/delete used here). For async
    contexts wrap calls in run_in_executor.

    Args:
        redis_client: redis.Redis (sync) compatible client.
        session_id: caller's session identifier — keyspace boundary.
        ttl_seconds: hash TTL refreshed on every write. Default 24h.
        key_prefix: customisable Redis key namespace.
    """

    DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24h

    def __init__(
        self,
        redis_client: Any,
        session_id: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = "ubp:citations:session",
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required for session-scoped store")
        self._redis = redis_client
        self._session_id = session_id
        self._ttl = int(ttl_seconds)
        self._key = f"{key_prefix}:{session_id}"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def redis_key(self) -> str:
        return self._key

    @staticmethod
    def _decode(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return None
        return str(value)

    def add(self, citation: Citation) -> Tuple[bool, Optional[str]]:
        """Add citation to session hash. Refreshes TTL.

        Returns (True, None) on success, (False, existing_id) if a citation
        with the same normalised DOI/URL/title is already in the session
        scope (cheap O(N) scan — sessions are bounded).
        """
        existing = self._find_duplicate(citation)
        if existing:
            return False, existing
        payload = json.dumps({
            **citation.to_dict(),
            "created_at": citation.created_at.isoformat(),
        })
        self._redis.hset(self._key, citation.id, payload)
        self._redis.expire(self._key, self._ttl)
        return True, None

    def get(self, citation_id: str) -> Optional[Citation]:
        raw = self._redis.hget(self._key, citation_id)
        decoded = self._decode(raw)
        if not decoded:
            return None
        try:
            return self._deserialize(json.loads(decoded))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "Malformed citation %s in session %s: %s",
                citation_id, self._session_id, e,
            )
            return None

    def list_all(self) -> List[Citation]:
        raw_map = self._redis.hgetall(self._key) or {}
        out: List[Citation] = []
        for _field, value in raw_map.items():
            decoded = self._decode(value)
            if not decoded:
                continue
            try:
                out.append(self._deserialize(json.loads(decoded)))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return out

    def delete(self, citation_id: str) -> bool:
        removed = self._redis.hdel(self._key, citation_id)
        return bool(removed)

    def clear(self) -> bool:
        """Drop the entire session-scoped citation set."""
        return bool(self._redis.delete(self._key))

    def count(self) -> int:
        try:
            return int(self._redis.hlen(self._key) or 0)
        except (TypeError, ValueError):
            return 0

    # -- helpers ------------------------------------------------------------

    def _find_duplicate(self, citation: Citation) -> Optional[str]:
        """Cheap dup detection over the session-scoped set."""
        norm_doi = (citation.doi or "").lower().strip() or None
        norm_url = self._normalize_url(citation.url) if citation.url else None
        norm_title = self._normalize_title(citation.title) if citation.title else None
        for existing in self.list_all():
            if existing.id == citation.id:
                return existing.id
            if norm_doi and existing.doi and existing.doi.lower().strip() == norm_doi:
                return existing.id
            if norm_url and existing.url and self._normalize_url(existing.url) == norm_url:
                return existing.id
            if norm_title and self._normalize_title(existing.title) == norm_title:
                return existing.id
        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        return re.sub(r"[?#].*$", "", (url or "").lower().rstrip("/"))

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", (title or "").lower().strip())

    @staticmethod
    def _deserialize(data: Dict[str, Any]) -> Citation:
        # Reuse the existing dict→Citation logic from CitationStore by
        # dispatching through a temporary instance method.
        return CitationStore.__dict__["_dict_to_citation"](
            CitationStore.__new__(CitationStore), data
        )


# ---------------------------------------------------------------------------
# fc3 (B-07 deep) — URL validation against session citation store
# ---------------------------------------------------------------------------
# Closes the loop opened by fa5 (synthesis prompt anti-URL guard) and fb2
# (Redis-backed session-scoped store). The synthesis output may still
# contain URLs hallucinated by the model despite the prompt guard; this
# module provides the programmatic validator that synthesis pipelines can
# use to detect and either remove or flag any URL not present in the
# citation_store of the active session.
#
# Wiring (intentionally NOT done in this commit — requires a live
# synthesis test cycle):
#   1. After synthesis returns, call
#      ``unknown = find_unknown_urls(synthesis_text, store)``
#   2. If ``unknown`` is non-empty, replace each URL with
#      "[fonte non disponibile]" or surface a warning.
#   3. Log a structured event ``[CITATION-GUARD] unknown_urls=N session=...``
#      so admins can monitor hallucination rate.

# Conservative URL regex: matches http(s)://… stopping at common
# delimiters. Does not try to parse markdown link syntax — those are
# handled by extracting the bare URL part.
_URL_REGEX = re.compile(
    r"https?://[^\s\)\]\}<>\"\'\,]+",
    re.IGNORECASE,
)


def extract_urls(text: str) -> List[str]:
    """Extract bare http(s) URLs from arbitrary text.

    Markdown links ``[label](url)`` work because the trailing ``)`` is in
    the delimiter set, so the URL portion is captured cleanly. Trailing
    punctuation (``.``, ``,``) commonly attached after a URL is stripped.
    """
    if not text:
        return []
    urls = []
    for raw in _URL_REGEX.findall(text):
        cleaned = raw.rstrip(".,;!?")
        if cleaned:
            urls.append(cleaned)
    return urls


def find_unknown_urls(
    text: str,
    store: "SessionScopedCitationStore",
) -> List[str]:
    """Return URLs in ``text`` that are NOT present in the session store.

    Comparison is done on the normalised URL form (lowercase, trailing
    slash and query/fragment stripped) used by ``_normalize_url`` so
    cosmetic differences do not produce false positives.

    Args:
        text: synthesis output to scrutinise.
        store: caller's SessionScopedCitationStore (already scoped to
            the active session_id).

    Returns:
        Sorted, deduplicated list of unknown URLs (in their original form).
    """
    found = extract_urls(text)
    if not found:
        return []
    known_norm = set()
    try:
        for citation in store.list_all():
            if citation.url:
                known_norm.add(store._normalize_url(citation.url))
    except Exception as exc:
        logger.warning(
            "[CITATION-GUARD] could not enumerate session store: %s — "
            "treating all extracted URLs as unknown (fail-safe)",
            exc,
        )
    unknown = []
    seen = set()
    for url in found:
        norm = store._normalize_url(url)
        if norm in known_norm or norm in seen:
            continue
        seen.add(norm)
        unknown.append(url)
    return sorted(unknown)


__all__.extend([
    "extract_urls",
    "find_unknown_urls",
])
