"""
citation_manager/adapter.py

Bridge Layer - Citation management and bibliography generation.

Provides:
- Citation storage with deduplication
- Multi-style formatting (APA, MLA, Chicago, IEEE, etc.)
- Validation and verification
- Cross-document citation tracking
- BibTeX export/import

Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
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
    CitationStyle,
    SourceType,
    CitationStatus,
    # Data classes
    Author,
    Citation,
    InlineCitation,
    Bibliography,
    ValidationResult,
    ExportResult,
    ImportResult,
    # Components
    CitationStore,
    CitationFormatter,
    BibTeXFormatter,
    CitationValidator,
    DuplicateDetector,
)

logger = logging.getLogger(__name__)


class CitationManagerAdapter:
    """
    Main adapter for citation management.
    
    Provides operations for:
    - Adding and managing citations
    - Formatting in various academic styles
    - Generating bibliographies
    - Validation and duplicate detection
    - Export/Import (BibTeX, JSON)
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
        self._store: Optional[CitationStore] = None
        self._formatter: Optional[CitationFormatter] = None
        self._bibtex_formatter: Optional[BibTeXFormatter] = None
        self._validator: Optional[CitationValidator] = None
        self._duplicate_detector: Optional[DuplicateDetector] = None
        
        # Config
        self._default_style = CitationStyle.APA
        self._persist_path: Optional[Path] = None
        
        # State
        self._initialized = False
    
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
        default_style: str = "apa",
        persist_path: Optional[str] = None,
        strict_validation: bool = False,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Initialize citation manager.
        
        Args:
            default_style: Default citation style
            persist_path: Path for persistent storage
            strict_validation: Enable strict validation mode
        """
        self._default_style = CitationStyle(default_style)
        self._strict_validation = strict_validation

        # Set persistence path
        if persist_path:
            self._persist_path = Path(persist_path)
        else:
            self._persist_path = self.module_path / "citations.json"

        # Initialize components
        self._store = CitationStore(persist_path=self._persist_path)
        self._formatter = CitationFormatter()
        self._bibtex_formatter = BibTeXFormatter()
        self._validator = CitationValidator()
        self._duplicate_detector = DuplicateDetector()

        self._initialized = True

        logger.info(f"citation_manager initialized with {default_style} style")

        return {
            "status": "initialized",
            "module": "citation_manager",
            "version": "1.0.0",
            "default_style": default_style,
            "citations_loaded": len(self._store.list_all()),
        }
    
    async def shutdown(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Shutdown citation manager."""
        self._initialized = False
        return {"status": "shutdown"}
    
    async def health_check(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Health check."""
        return {
            "module": "citation_manager",
            "version": "1.0.0",
            "status": "healthy" if self._initialized else "not_initialized",
            "citations_count": len(self._store.list_all()) if self._store else 0,
        }
    
    # ========================================================================
    # Citation CRUD Operations
    # ========================================================================
    
    async def add_citation(
        self,
        title: str,
        authors: Optional[List[Dict[str, str]]] = None,
        year: Optional[str] = None,
        source_type: str = "other",
        journal: Optional[str] = None,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
        publisher: Optional[str] = None,
        doi: Optional[str] = None,
        url: Optional[str] = None,
        isbn: Optional[str] = None,
        collection: Optional[str] = None,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        relevance_score: float = 0.0,
        excerpt: str = "",
        tags: Optional[List[str]] = None,
        section_id: Optional[str] = None,
        validate: bool = True,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a new citation.
        
        Args:
            title: Citation title
            authors: List of author dicts with last_name, first_name
            year: Publication year
            source_type: Type of source
            journal: Journal name
            volume, issue, pages: Publication details
            publisher: Publisher name
            doi: Digital Object Identifier
            url: URL
            isbn: ISBN for books
            collection: RAG collection name
            document_id: Source document ID
            chunk_id: Source chunk ID
            relevance_score: Relevance from retrieval
            excerpt: Relevant text excerpt
            tags: Tags for organization
            section_id: Associated section
            validate: Run validation
        
        Returns:
            Dict with citation result
        """
        if not self._initialized:
            await self.initialize()
        
        # Build authors
        author_list = []
        if authors:
            for a in authors:
                # Bug 2 Fix: Handle both string and dict authors
                if isinstance(a, str):
                    # Simple string author
                    author_list.append(Author(last_name=a))
                elif isinstance(a, dict):
                    # Bug 4 Fix: Check for "name" field as fallback
                    if "name" in a and "last_name" not in a:
                        # Parse "name" field into last_name and first_name
                        # Note: This uses simple rsplit on last space, assuming Western naming
                        # conventions. For complex names (e.g., "Ludwig van Beethoven"),
                        # prefer providing explicit "last_name" and "first_name" fields.
                        parts = a["name"].rsplit(" ", 1)
                        last_name = parts[-1] if parts else ""
                        first_name = parts[0] if len(parts) > 1 else ""
                        author_list.append(Author(
                            last_name=last_name,
                            first_name=first_name,
                            middle_name=a.get("middle_name", ""),
                        ))
                    else:
                        author_list.append(Author(
                            last_name=a.get("last_name", ""),
                            first_name=a.get("first_name", ""),
                            middle_name=a.get("middle_name", ""),
                        ))
        
        # Create citation
        citation = Citation(
            id=str(uuid.uuid4()),
            title=title,
            authors=author_list,
            year=str(year) if year is not None else None,  # Bug 3 Fix: Convert to string
            source_type=SourceType(source_type),
            journal=journal,
            volume=volume,
            issue=issue,
            pages=pages,
            publisher=publisher,
            doi=doi,
            url=url,
            isbn=isbn,
            collection=collection,
            document_id=document_id,
            chunk_id=chunk_id,
            relevance_score=relevance_score,
            excerpt=excerpt,
            tags=tags or [],
            created_at=datetime.utcnow(),
        )
        
        # Validate
        validation = None
        if validate:
            validation = self._validator.validate(citation)
            if not validation.valid and self._strict_validation:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "validation": validation.to_dict(),
                }

        # Store
        success, duplicate_id = self._store.add(citation, check_duplicates=True)

        if not success and duplicate_id:
            return {
                "success": True,
                "citation_id": duplicate_id,
                "is_duplicate": True,
                "validation": validation.to_dict() if validation else None,
            }

        return {
            "success": True,
            "citation_id": citation.id,
            "is_duplicate": False,
            "validation": validation.to_dict() if validation else None,
        }
    
    async def add_from_document(
        self,
        document: Dict[str, Any],
        section_id: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add citation from a retrieved document.
        
        Args:
            document: Document dict with metadata
            section_id: Associated section
        
        Returns:
            Dict with citation result
        """
        # Extract metadata
        metadata = document.get("metadata", {})
        
        # Parse authors if string
        authors = []
        author_str = metadata.get("authors", metadata.get("author", ""))
        if isinstance(author_str, str) and author_str:
            # Simple parsing: split by comma or "and"
            for name in author_str.replace(" and ", ",").split(","):
                name = name.strip()
                if name:
                    parts = name.split()
                    if len(parts) >= 2:
                        authors.append({
                            "first_name": " ".join(parts[:-1]),
                            "last_name": parts[-1],
                        })
                    else:
                        authors.append({"last_name": name, "first_name": ""})
        elif isinstance(author_str, list):
            authors = author_str
        
        return await self.add_citation(
            title=metadata.get("title", document.get("title", "Untitled")),
            authors=authors,
            year=metadata.get("year", metadata.get("date", "")[:4] if metadata.get("date") else None),
            source_type=metadata.get("source_type", "rag_document"),
            journal=metadata.get("journal"),
            doi=metadata.get("doi"),
            url=metadata.get("url", metadata.get("source")),
            collection=document.get("collection"),
            document_id=document.get("id", document.get("document_id")),
            chunk_id=document.get("chunk_id"),
            relevance_score=document.get("score", document.get("relevance_score", 0.0)),
            excerpt=document.get("content", "")[:500],
            section_id=section_id,
        )
    
    async def get_citation(
        self,
        citation_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get a citation by ID."""
        if not self._initialized:
            await self.initialize()
        
        citation = self._store.get(citation_id)
        
        if not citation:
            return {"success": False, "error": "Citation not found"}
        
        return {
            "success": True,
            "citation": citation.to_dict(),
        }
    
    async def update_citation(
        self,
        citation_id: str,
        updates: Dict[str, Any],
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update a citation."""
        if not self._initialized:
            await self.initialize()
        
        citation = self._store.get(citation_id)
        if not citation:
            return {"success": False, "error": "Citation not found"}
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(citation, key):
                if key == "authors" and isinstance(value, list):
                    citation.authors = [
                        Author(
                            last_name=a.get("last_name", ""),
                            first_name=a.get("first_name", ""),
                        )
                        for a in value
                    ]
                elif key == "source_type":
                    citation.source_type = SourceType(value)
                elif key == "status":
                    citation.status = CitationStatus(value)
                else:
                    setattr(citation, key, value)
        
        self._store.update(citation)
        
        return {"success": True, "citation_id": citation_id}
    
    async def delete_citation(
        self,
        citation_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete a citation."""
        if not self._initialized:
            await self.initialize()
        
        success = self._store.delete(citation_id)
        
        return {
            "success": success,
            "error": None if success else "Citation not found",
        }
    
    async def list_citations(
        self,
        source_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        section_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List citations with filtering.
        
        Args:
            source_type: Filter by type
            tags: Filter by tags
            section_id: Filter by section
            document_id: Filter by document
            query: Search query
            limit: Maximum results
        """
        if not self._initialized:
            await self.initialize()

        citations = self._store.search(
            query=query,
            source_type=SourceType(source_type) if source_type else None,
            tags=tags,
            section_id=section_id,
            document_id=document_id,
            limit=limit,
        )

        return {
            "success": True,
            "citations": [c.to_dict() for c in citations],
            "count": len(citations),
        }

    # ========================================================================
    # Formatting Operations
    # ========================================================================
    
    async def format_citation(
        self,
        citation_id: str,
        style: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Format a citation in specified style.
        
        Args:
            citation_id: Citation ID
            style: Citation style (apa, mla, chicago, ieee, etc.)
        """
        if not self._initialized:
            await self.initialize()
        
        citation = self._store.get(citation_id)
        if not citation:
            return {"success": False, "error": "Citation not found"}
        
        style_enum = CitationStyle(style) if style else self._default_style
        formatted = self._formatter.format_citation(citation, style_enum)
        
        return {
            "success": True,
            "formatted": formatted,
            "style": style_enum.value,
        }
    
    async def format_inline(
        self,
        citation_id: str,
        number: int = 1,
        page: Optional[str] = None,
        style: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Format inline citation reference.
        
        Args:
            citation_id: Citation ID
            number: Citation number for numeric styles
            page: Specific page reference
            style: Citation style
        """
        if not self._initialized:
            await self.initialize()
        
        citation = self._store.get(citation_id)
        if not citation:
            return {"success": False, "error": "Citation not found"}
        
        style_enum = CitationStyle(style) if style else self._default_style
        inline = self._formatter.format_inline(citation, style_enum, number, page)
        
        # Update usage count
        citation.usage_count += 1
        self._store.update(citation)
        
        return {
            "success": True,
            "inline": inline,
            "style": style_enum.value,
        }
    
    async def generate_bibliography(
        self,
        citation_ids: Optional[List[str]] = None,
        section_id: Optional[str] = None,
        document_id: Optional[str] = None,
        style: Optional[str] = None,
        title: str = "Bibliografia",
        numbered: bool = True,
        sort_by: str = "author",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate formatted bibliography.
        
        Args:
            citation_ids: Specific citations to include
            section_id: Include citations from section
            document_id: Include citations from document
            style: Citation style
            title: Bibliography title
            numbered: Number entries
            sort_by: Sort order (author, year, order)
        """
        if not self._initialized:
            await self.initialize()
        
        # Get citations
        if citation_ids:
            citations = [self._store.get(cid) for cid in citation_ids]
            citations = [c for c in citations if c]
        elif section_id or document_id:
            citations = self._store.search(section_id=section_id, document_id=document_id)
        else:
            citations = self._store.list_all()

        if not citations:
            return {
                "success": True,
                "bibliography": "",
                "count": 0,
            }

        # Sort citations
        style_enum = CitationStyle(style) if style else self._default_style
        if sort_by == "author":
            citations = sorted(citations, key=lambda c: c.authors[0].last_name if c.authors else "")
        elif sort_by == "year":
            citations = sorted(citations, key=lambda c: c.year or "9999")

        # Format bibliography entries
        entries = []
        for i, citation in enumerate(citations, 1):
            formatted = self._formatter.format(citation, style_enum)
            if numbered:
                entries.append(f"[{i}] {formatted}")
            else:
                entries.append(formatted)

        bibliography_text = f"# {title}\n\n" + "\n\n".join(entries)

        return {
            "success": True,
            "bibliography": bibliography_text,
            "style": style_enum.value,
            "count": len(citations),
        }
    
    # ========================================================================
    # Validation Operations
    # ========================================================================
    
    async def validate_citation(
        self,
        citation_id: str,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validate a citation."""
        if not self._initialized:
            await self.initialize()
        
        citation = self._store.get(citation_id)
        if not citation:
            return {"success": False, "error": "Citation not found"}
        
        result = self._validator.validate(citation)
        
        return {
            "success": True,
            "validation": result.to_dict(),
        }
    
    async def validate_all(
        self,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validate all citations."""
        if not self._initialized:
            await self.initialize()

        citations = self._store.list_all()
        results = self._validator.validate_all(citations)

        valid_count = sum(1 for r in results.values() if r.valid)

        return {
            "success": True,
            "total": len(results),
            "valid": valid_count,
            "invalid": len(results) - valid_count,
            "details": {cid: r.to_dict() for cid, r in results.items() if not r.valid},
        }

    async def find_duplicates(
        self,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Find duplicate citations."""
        if not self._initialized:
            await self.initialize()

        citations = self._store.list_all()
        duplicates = self._duplicate_detector.find_duplicates(citations)

        return {
            "success": True,
            "duplicates": [
                {"id1": d[0], "id2": d[1], "reason": d[2]}
                for d in duplicates
            ],
            "count": len(duplicates),
        }

    # ========================================================================
    # Export/Import Operations
    # ========================================================================

    async def export_bibtex(
        self,
        citation_ids: Optional[List[str]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export citations as BibTeX."""
        if not self._initialized:
            await self.initialize()

        if citation_ids:
            citations = [self._store.get(cid) for cid in citation_ids]
            citations = [c for c in citations if c]
        else:
            citations = self._store.list_all()

        # Format each citation as BibTeX
        bibtex_entries = [self._bibtex_formatter.format(c) for c in citations]
        bibtex = "\n\n".join(bibtex_entries)

        return {
            "success": True,
            "format": "bibtex",
            "content": bibtex,
            "count": len(citations),
        }

    async def export_json(
        self,
        citation_ids: Optional[List[str]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export citations as JSON."""
        if not self._initialized:
            await self.initialize()

        if citation_ids:
            citations = [self._store.get(cid) for cid in citation_ids]
            citations = [c for c in citations if c]
        else:
            citations = self._store.list_all()

        return {
            "success": True,
            "format": "json",
            "citations": [c.to_dict() for c in citations],
            "count": len(citations),
        }

    async def get_stats(
        self,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get citation statistics."""
        if not self._initialized:
            await self.initialize()

        citations = self._store.list_all()

        # Count by source type
        by_type = {}
        for c in citations:
            t = c.source_type.value
            by_type[t] = by_type.get(t, 0) + 1

        stats = {
            "total_citations": len(citations),
            "by_source_type": by_type,
            "default_style": self._default_style.value,
        }

        return {
            "success": True,
            "stats": stats,
        }

    async def clear_all(
        self,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Clear all citations."""
        if not self._initialized:
            await self.initialize()

        citations = self._store.list_all()
        count = 0
        for c in citations:
            if self._store.delete(c.id):
                count += 1

        return {
            "success": True,
            "deleted_count": count,
        }
