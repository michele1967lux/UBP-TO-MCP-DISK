"""
document_renderer/adapter.py

Bridge Layer - Multi-format document rendering.

Provides:
- PDF native rendering
- DOCX with templates
- PPTX presentations
- Chart and table generation
- Image embedding

Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
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
    OutputFormat,
    PageSize,
    Orientation,
    ChartType,
    TableStyle,
    ImagePosition,
    # Style
    Margins,
    FontConfig,
    StyleConfig,
    # Content
    TableCell,
    TableData,
    ChartData,
    ImageData,
    SectionContent,
    DocumentContent,
    SlideContent,
    PresentationContent,
    # Options
    PdfOptions,
    DocxOptions,
    PptxOptions,
    XlsxOptions,
    # Results
    RenderResult,
    ChartResult,
    MultiRenderResult,
    TemplateInfo,
    # Builders
    ChartBuilder,
    TableBuilder,
    # Renderers
    PdfRenderer,
    DOCXRenderer,
    PPTXRenderer,
)

# Aliases for adapter compatibility
DocxRenderer = DOCXRenderer
PptxRenderer = PPTXRenderer

logger = logging.getLogger(__name__)


class DocumentRendererAdapter:
    """
    Main adapter for document rendering.
    
    Provides operations for:
    - Multi-format document rendering (PDF, DOCX, PPTX)
    - Chart generation
    - Table formatting
    - Template management
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
        self._chart_builder: Optional[ChartBuilder] = None
        self._table_builder: Optional[TableBuilder] = None
        self._pdf_renderer: Optional[PdfRenderer] = None
        self._docx_renderer: Optional[DocxRenderer] = None
        self._pptx_renderer: Optional[PptxRenderer] = None
        
        # Templates
        self._templates_dir: Optional[Path] = None
        self._templates: Dict[str, TemplateInfo] = {}
        
        # State
        self._initialized = False
        
        # Default style
        self._default_style = StyleConfig()
    
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
        default_style: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Initialize the document renderer.
        
        Args:
            templates_dir: Directory with custom templates
            default_style: Default styling configuration
        """
        # Set templates directory
        if templates_dir:
            self._templates_dir = Path(templates_dir)
        elif (self.module_path / "templates").exists():
            self._templates_dir = self.module_path / "templates"
        
        # Load templates
        if self._templates_dir:
            self._load_templates()
        
        # Set default style
        if default_style:
            self._default_style = self._build_style_config(default_style)
        
        # Initialize components
        self._chart_builder = ChartBuilder()
        self._table_builder = TableBuilder()

        self._pdf_renderer = PdfRenderer(style=self._default_style)
        self._docx_renderer = DocxRenderer(style=self._default_style)
        self._pptx_renderer = PptxRenderer(style=self._default_style)
        
        self._initialized = True
        
        logger.info("document_renderer initialized")
        
        return {
            "status": "initialized",
            "module": "document_renderer",
            "version": "1.0.0",
            "templates_loaded": len(self._templates),
            "supported_formats": ["pdf", "docx", "pptx", "xlsx", "md", "html"],
        }
    
    async def shutdown(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Shutdown the document renderer."""
        self._initialized = False
        return {"status": "shutdown"}
    
    async def health_check(self, ctx: Any = None, **kwargs) -> Dict[str, Any]:
        """Health check."""
        return {
            "module": "document_renderer",
            "version": "1.0.0",
            "status": "healthy" if self._initialized else "not_initialized",
            "templates_count": len(self._templates),
        }
    
    def _load_templates(self) -> None:
        """Load available templates from directory."""
        if not self._templates_dir or not self._templates_dir.exists():
            return
        
        for format_dir in self._templates_dir.iterdir():
            if format_dir.is_dir():
                format_name = format_dir.name.lower()
                for template_file in format_dir.glob("*"):
                    if template_file.suffix in ['.docx', '.pptx', '.xlsx']:
                        template_id = f"{format_name}_{template_file.stem}"
                        self._templates[template_id] = TemplateInfo(
                            id=template_id,
                            name=template_file.stem.replace("_", " ").title(),
                            description=f"{format_name.upper()} template",
                            format=OutputFormat(format_name),
                            path=str(template_file),
                        )
    
    # ========================================================================
    # Core Rendering Operations
    # ========================================================================

    async def render_auto(
        self,
        content: Union[Dict, "DocumentContent"],
        format: str = "pdf",
        options: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Auto-select render method based on format, with fallback chain."""
        render_methods = {
            "pdf": self.render_pdf,
            "docx": self.render_docx,
            "pptx": self.render_pptx,
            "markdown": self.render_markdown,
        }
        fallback_order = ["docx", "markdown"]
        
        method = render_methods.get(format, self.render_pdf)
        
        async def _try_render(fmt, m):
            render_kwargs = {"content": content, "ctx": ctx}
            if fmt != "markdown":
                render_kwargs["options"] = options
            render_kwargs.update(kwargs)
            try:
                return await m(**render_kwargs)
            except Exception:
                # Options may be incompatible — retry without options
                render_kwargs_no_opts = {"content": content, "ctx": ctx}
                render_kwargs_no_opts.update(kwargs)
                return await m(**render_kwargs_no_opts)
        
        result = await _try_render(format, method)
        
        # Fallback if primary format failed
        if not result.get("success") and format not in fallback_order[:1]:
            for fb_format in fallback_order:
                if fb_format == format:
                    continue
                fb_method = render_methods[fb_format]
                result = await _try_render(fb_format, fb_method)
                if result.get("success"):
                    logger.info(f"render_auto: {format} failed, fell back to {fb_format}")
                    break
        
        # Add filename if not present
        if result.get("success") and "filename" not in result:
            title = content.get("title", "document") if isinstance(content, dict) else "document"
            ext = result.get("format", format)
            result["filename"] = f"{title[:50]}.{ext}"
        
        return result
    
    async def render_pdf(
        self,
        content: Union[Dict, DocumentContent],
        options: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render document to PDF.
        
        Args:
            content: Document content
            options: PDF rendering options
            output_path: Optional path to save file
        
        Returns:
            Dict with render result
        """
        if not self._initialized:
            await self.initialize()
        
        # Build content object
        doc_content = self._build_document_content(content)
        
        # Build options
        pdf_options = self._build_pdf_options(options)
        
        # Render
        result = await self._pdf_renderer.render(doc_content, pdf_options)
        
        # Save if path provided
        if output_path and result.success and result.content:
            Path(output_path).write_bytes(result.content)
        
        return {
            "success": result.success,
            "format": "pdf",
            "pages": result.page_count,
            "file_size": result.file_size,
            "render_time_ms": result.time_ms,
            "content": result.content_base64 if not output_path else None,
            "output_path": output_path,
            "error": result.error,
        }
    
    async def render_docx(
        self,
        content: Union[Dict, DocumentContent],
        options: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        output_path: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render document to DOCX.
        
        Args:
            content: Document content
            options: DOCX rendering options
            template_id: Optional template to use
            output_path: Optional path to save file
        
        Returns:
            Dict with render result
        """
        if not self._initialized:
            await self.initialize()
        
        # Build content object
        doc_content = self._build_document_content(content)
        
        # Build options
        docx_options = self._build_docx_options(options)
        
        # Apply template
        if template_id and template_id in self._templates:
            docx_options.template_path = self._templates[template_id].path
        
        # Render
        result = await self._docx_renderer.render(doc_content, docx_options)
        
        # Save if path provided
        if output_path and result.success and result.content:
            Path(output_path).write_bytes(result.content)
        
        return {
            "success": result.success,
            "format": "docx",
            "pages": result.page_count,
            "file_size": result.file_size,
            "render_time_ms": result.time_ms,
            "content": result.content_base64 if not output_path else None,
            "output_path": output_path,
            "error": result.error,
        }
    
    async def render_pptx(
        self,
        content: Union[Dict, PresentationContent],
        options: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        output_path: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render presentation to PPTX.
        
        Args:
            content: Presentation content
            options: PPTX rendering options
            template_id: Optional template to use
            output_path: Optional path to save file
        
        Returns:
            Dict with render result
        """
        if not self._initialized:
            await self.initialize()
        
        # Build content object
        pres_content = self._build_presentation_content(content)
        
        # Build options
        pptx_options = self._build_pptx_options(options)
        
        # Apply template
        if template_id and template_id in self._templates:
            pptx_options.template_path = self._templates[template_id].path
        
        # Render
        result = await self._pptx_renderer.render(pres_content, pptx_options)
        
        # Save if path provided
        if output_path and result.success and result.content:
            Path(output_path).write_bytes(result.content)
        
        return {
            "success": result.success,
            "format": "pptx",
            "slides": result.page_count,
            "file_size": result.file_size,
            "render_time_ms": result.time_ms,
            "content": result.content_base64 if not output_path else None,
            "output_path": output_path,
            "error": result.error,
        }
    
    async def render_multi(
        self,
        content: Union[Dict, DocumentContent],
        formats: List[str],
        options: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render document to multiple formats in parallel.
        
        Args:
            content: Document content
            formats: List of formats to render (pdf, docx, pptx)
            options: Dict of format-specific options
            output_dir: Optional directory to save files
        
        Returns:
            Dict with results per format
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        options = options or {}
        
        # Build content
        doc_content = self._build_document_content(content)
        
        # Bug 9 Fix: Map formats to tasks with a dict to avoid index issues
        format_to_task = {}
        for fmt in formats:
            fmt_lower = fmt.lower()
            fmt_options = options.get(fmt_lower, {})
            
            output_path = None
            if output_dir:
                output_path = str(Path(output_dir) / f"{doc_content.title}.{fmt_lower}")
            
            if fmt_lower == "pdf":
                format_to_task[fmt] = self.render_pdf(content, fmt_options, output_path)
            elif fmt_lower == "docx":
                format_to_task[fmt] = self.render_docx(content, fmt_options, None, output_path)
            elif fmt_lower == "pptx":
                # Convert to presentation if needed
                pres_content = self._document_to_presentation(doc_content)
                format_to_task[fmt] = self.render_pptx(pres_content, fmt_options, None, output_path)
            # If format is not recognized (e.g., markdown), skip it
        
        # Execute in parallel
        if format_to_task:
            results = await asyncio.gather(*format_to_task.values(), return_exceptions=True)
            
            # Build response
            format_results = {}
            for i, fmt in enumerate(format_to_task.keys()):
                if isinstance(results[i], Exception):
                    format_results[fmt] = {"success": False, "error": str(results[i])}
                else:
                    format_results[fmt] = results[i]
        else:
            format_results = {}
        
        total_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": all(r.get("success", False) for r in format_results.values()),
            "formats_rendered": len([r for r in format_results.values() if r.get("success")]),
            "results": format_results,
            "total_time_ms": total_time,
        }
    
    async def render_markdown(
        self,
        content: Union[Dict, DocumentContent],
        output_path: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render document to Markdown.
        
        Args:
            content: Document content
            output_path: Optional path to save file
        
        Returns:
            Dict with markdown content
        """
        if not self._initialized:
            await self.initialize()
        
        doc_content = self._build_document_content(content)
        
        # Build markdown
        lines = []
        
        # Title
        lines.append(f"# {doc_content.title}\n")
        
        # Bug 7 Fix: Use correct field names
        if doc_content.authors:
            lines.append(f"*Autore: {', '.join(doc_content.authors)}*\n")
        
        if doc_content.date:
            lines.append(f"*Data: {doc_content.date}*\n")
        lines.append("---\n")
        
        # Sections
        for section in doc_content.sections:
            heading = "#" * (section.level + 1)
            lines.append(f"\n{heading} {section.title}\n")
            lines.append(section.content)
            lines.append("")
            
            # Tables
            for table in section.tables:
                lines.append(self._table_builder.to_markdown(table))
                lines.append("")
        
        # Bibliography
        if doc_content.bibliography:
            lines.append("\n## Bibliografia\n")
            lines.append(doc_content.bibliography)
        
        markdown = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(markdown, encoding="utf-8")
        
        return {
            "success": True,
            "format": "md",
            "content": markdown if not output_path else None,
            "output_path": output_path,
        }
    
    # ========================================================================
    # Chart Operations
    # ========================================================================
    
    async def render_chart(
        self,
        chart_data: Union[Dict, ChartData],
        output_format: str = "png",
        output_path: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render a chart.
        
        Args:
            chart_data: Chart data and configuration
            output_format: Output format (png, svg)
            output_path: Optional path to save file
        
        Returns:
            Dict with chart result
        """
        if not self._initialized:
            await self.initialize()
        
        chart = self._build_chart_data(chart_data)
        
        result = await self._chart_builder.generate(chart, output_format)
        
        if output_path and result.success and result.image_bytes:
            Path(output_path).write_bytes(result.image_bytes)
        
        return {
            "success": result.success,
            "format": result.image_format,
            "width": result.width,
            "height": result.height,
            "content": result.image_bytes if not output_path else None,
            "output_path": output_path,
            "error": result.error,
        }
    
    async def render_charts_batch(
        self,
        charts: List[Union[Dict, ChartData]],
        output_format: str = "png",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render multiple charts.
        
        Args:
            charts: List of chart data
            output_format: Output format
        
        Returns:
            Dict with chart results
        """
        if not self._initialized:
            await self.initialize()
        
        chart_objects = [self._build_chart_data(c) for c in charts]
        results = await self._chart_builder.generate_batch(chart_objects, output_format)
        
        return {
            "success": all(r.success for r in results),
            "charts": [
                {
                    "success": r.success,
                    "chart_type": r.chart_type.value,
                    "content": r.image_bytes,
                    "error": r.error,
                }
                for r in results
            ],
        }
    
    # ========================================================================
    # Table Operations
    # ========================================================================
    
    async def render_table(
        self,
        table_data: Union[Dict, TableData],
        output_format: str = "html",
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render a table to specified format.
        
        Args:
            table_data: Table data
            output_format: Output format (html, markdown)
        
        Returns:
            Dict with rendered table
        """
        if not self._initialized:
            await self.initialize()
        
        table = self._build_table_data(table_data)
        
        if output_format == "markdown" or output_format == "md":
            content = self._table_builder.to_markdown(table)
        else:
            content = self._table_builder.to_html(table)
        
        return {
            "success": True,
            "format": output_format,
            "content": content,
        }
    
    # ========================================================================
    # Template Operations
    # ========================================================================
    
    async def list_templates(
        self,
        format_filter: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List available templates.
        
        Args:
            format_filter: Optional format filter
        
        Returns:
            Dict with templates list
        """
        if not self._initialized:
            await self.initialize()
        
        templates = list(self._templates.values())
        
        if format_filter:
            templates = [t for t in templates if t.format.value == format_filter.lower()]
        
        return {
            "success": True,
            "templates": [t.to_dict() for t in templates],
            "count": len(templates),
        }
    
    async def apply_template(
        self,
        template_id: str,
        content: Union[Dict, DocumentContent],
        output_path: Optional[str] = None,
        ctx: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Render using a specific template.
        
        Args:
            template_id: Template identifier
            content: Document content
            output_path: Optional output path
        
        Returns:
            Dict with render result
        """
        if template_id not in self._templates:
            return {
                "success": False,
                "error": f"Template '{template_id}' not found",
            }
        
        template = self._templates[template_id]
        
        if template.format == OutputFormat.DOCX:
            return await self.render_docx(content, template_id=template_id, output_path=output_path)
        elif template.format == OutputFormat.PPTX:
            return await self.render_pptx(content, template_id=template_id, output_path=output_path)
        else:
            return {
                "success": False,
                "error": f"Template format '{template.format}' not supported",
            }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _build_document_content(self, data: Union[Dict, DocumentContent]) -> DocumentContent:
        """Build DocumentContent from dict or pass through."""
        if isinstance(data, DocumentContent):
            return data
        
        sections = []
        for s in data.get("sections", []):
            sections.append(SectionContent(
                id=s.get("id", f"section_{len(sections)}"),
                title=s.get("title", ""),
                content=s.get("content", ""),
                level=s.get("level", 1),
                tables=[self._build_table_data(t) for t in s.get("tables", [])],
                charts=[self._build_chart_data(c) for c in s.get("charts", [])],
                images=[self._build_image_data(i) for i in s.get("images", [])],
                page_break_before=s.get("page_break_before", False),
            ))
        
        return DocumentContent(
            title=data.get("title", "Untitled"),
            subtitle=data.get("subtitle", ""),
            authors=[data.get("author")] if data.get("author") else data.get("authors", []),
            date=data.get("date"),
            abstract=data.get("abstract", ""),
            sections=sections,
            bibliography=data.get("bibliography", []),
            appendices=data.get("appendices", []),
            metadata=data.get("metadata", {}),
        )
    
    def _build_presentation_content(self, data: Union[Dict, PresentationContent]) -> PresentationContent:
        """Build PresentationContent from dict."""
        if isinstance(data, PresentationContent):
            return data
        
        slides = []
        for s in data.get("slides", []):
            slides.append(SlideContent(
                title=s.get("title", ""),
                content=s.get("content", ""),
                subtitle=s.get("subtitle"),
                notes=s.get("notes"),
                chart=self._build_chart_data(s.get("chart")) if s.get("chart") else None,
                table=self._build_table_data(s.get("table")) if s.get("table") else None,
                image=self._build_image_data(s.get("image")) if s.get("image") else None,
                layout=s.get("layout", "content"),
            ))
        
        return PresentationContent(
            title=data.get("title", "Untitled"),
            subtitle=data.get("subtitle", ""),
            author=data.get("author", ""),
            date=data.get("date"),
            slides=slides,
            theme=data.get("theme", "professional"),
            metadata=data.get("metadata", {}),
        )
    
    def _build_chart_data(self, data: Union[Dict, ChartData, None]) -> Optional[ChartData]:
        """Build ChartData from dict."""
        if data is None:
            return None
        if isinstance(data, ChartData):
            return data
        
        return ChartData(
            chart_type=ChartType(data.get("chart_type", data.get("type", "bar"))),
            title=data.get("title", ""),
            labels=data.get("labels", []),
            datasets=data.get("datasets", []),
            x_label=data.get("x_label"),
            y_label=data.get("y_label"),
            colors=data.get("colors"),
            show_legend=data.get("show_legend", True),
            show_grid=data.get("show_grid", True),
            width=data.get("width", 600),
            height=data.get("height", 400),
        )
    
    def _build_table_data(self, data: Union[Dict, TableData, None]) -> Optional[TableData]:
        """Build TableData from dict."""
        if data is None:
            return None
        if isinstance(data, TableData):
            return data
        
        return TableData(
            headers=data.get("headers", []),
            rows=data.get("rows", []),
            caption=data.get("caption"),
            style=TableStyle(data.get("style", "professional")),
            totals_row=data.get("totals_row", False),
            totals_columns=data.get("totals_columns", []),
        )
    
    def _build_image_data(self, data: Union[Dict, ImageData, None]) -> Optional[ImageData]:
        """Build ImageData from dict."""
        if data is None:
            return None
        if isinstance(data, ImageData):
            return data
        
        return ImageData(
            source=data.get("source", data.get("path", data.get("url", ""))),
            width=data.get("width"),
            height=data.get("height"),
            caption=data.get("caption"),
            position=ImagePosition(data.get("position", "center")),
        )
    
    def _build_style_config(self, data: Dict[str, Any]) -> StyleConfig:
        """Build StyleConfig from dict."""
        if not data:
            return self._default_style
        
        return StyleConfig(
            page_size=PageSize(data.get("page_size", "A4")),
            orientation=Orientation(data.get("orientation", "portrait")),
            primary_color=data.get("primary_color", "#1F4E79"),
            secondary_color=data.get("secondary_color", "#4472C4"),
            accent_color=data.get("accent_color", "#ED7D31"),
            table_style=TableStyle(data.get("table_style", "professional")),
            header_text=data.get("header_text"),
            footer_text=data.get("footer_text"),
            include_page_numbers=data.get("include_page_numbers", True),
        )
    
    def _build_pdf_options(self, data: Optional[Dict[str, Any]]) -> PdfOptions:
        """Build PdfOptions from dict."""
        if not data:
            return PdfOptions()
        
        return PdfOptions(
            page_size=PageSize(data.get("page_size", "A4")),
            orientation=Orientation(data.get("orientation", "portrait")),
            include_toc=data.get("include_toc", True),
            include_cover=data.get("include_cover", True),
            include_page_numbers=data.get("include_page_numbers", True),
            include_bookmarks=data.get("include_bookmarks", True),
            encrypt=data.get("encrypt", False),
            password=data.get("password"),
            watermark_text=data.get("watermark_text"),
            font_family=data.get("font_family", "Helvetica"),
            font_size=data.get("font_size", 11),
        )
    
    def _build_docx_options(self, data: Optional[Dict[str, Any]]) -> DocxOptions:
        """Build DocxOptions from dict."""
        if not data:
            return DocxOptions()
        
        return DocxOptions(
            template_path=data.get("template_path"),
            page_size=PageSize(data.get("page_size", "A4")),
            orientation=Orientation(data.get("orientation", "portrait")),
            include_toc=data.get("include_toc", True),
            include_cover=data.get("include_cover", True),
            include_page_numbers=data.get("include_page_numbers", True),
            include_bibliography=data.get("include_bibliography", True),
            font_family=data.get("font_family", "Calibri"),
            font_size=data.get("font_size", 11),
            line_spacing=data.get("line_spacing", 1.15),
            header_text=data.get("header_text"),
            footer_text=data.get("footer_text"),
        )
    
    def _build_pptx_options(self, data: Optional[Dict[str, Any]]) -> PptxOptions:
        """Build PptxOptions from dict."""
        if not data:
            return PptxOptions()
        
        return PptxOptions(
            template_path=data.get("template_path"),
            width_inches=data.get("width_inches", 13.333),
            height_inches=data.get("height_inches", 7.5),
            include_title_slide=data.get("include_title_slide", True),
            include_toc_slide=data.get("include_toc_slide", False),
            include_summary_slide=data.get("include_summary_slide", False),
            include_speaker_notes=data.get("include_speaker_notes", True),
            theme=data.get("theme", "professional"),
        )
    
    def _document_to_presentation(self, doc: DocumentContent) -> PresentationContent:
        """Convert document content to presentation."""
        slides = []
        
        for section in doc.sections:
            slide = SlideContent(
                title=section.title,
                content=section.content[:500] if len(section.content) > 500 else section.content,
                notes=section.content if len(section.content) > 500 else None,
                chart=section.charts[0] if section.charts else None,
                table=section.tables[0] if section.tables else None,
                layout="chart" if section.charts else ("table" if section.tables else "content"),
            )
            slides.append(slide)
        
        # Bug 9 Fix: Use correct fields - authors instead of author, no style field
        return PresentationContent(
            title=doc.title,
            slides=slides,
            author=", ".join(doc.authors) if doc.authors else "",
            # style field doesn't exist in PresentationContent
        )
