"""
RAG Orchestrator Renderers Package (v2.5.1 - FEAT-ARTIFACT-002)

This package contains renderers for converting report content
to various output formats.

Components:
- Base classes: FormatRenderer, OutputFormat, RenderContext, RenderResult
- DocxRenderer: Converts Markdown to Microsoft Word format
- MarkdownRenderer: Outputs clean Markdown
- CsvRenderer: Exports tabular data to CSV
- ExcelRenderer: Exports tabular data to Excel XLSX with formatting
- PptxRenderer: Creates PowerPoint presentations

Usage:
    from renderers import DocxRenderer, ExcelRenderer, PptxRenderer
    from renderers.base import RenderContext, OutputFormat

    context = RenderContext(title="My Report", content="...")
    renderer = DocxRenderer()
    result = renderer.render(context)
"""

# Base classes and enums
from .base import (
    FormatRenderer,
    OutputFormat,
    WorkerMode,
    PostProcessType,
    RenderContext,
    RenderResult,
    DataExtractor,
    TableDataProcessor,
)

# Document renderers
from .docx_renderer import DocxRenderer, DocxStyle, MarkdownRenderer

# Tabular data renderers
from .csv_renderer import CsvRenderer, ExcelRenderer

# Presentation renderer
from .pptx_renderer import PptxRenderer, SlideType, SlideContent

__all__ = [
    # Base classes
    "FormatRenderer",
    "OutputFormat",
    "WorkerMode",
    "PostProcessType",
    "RenderContext",
    "RenderResult",
    "DataExtractor",
    "TableDataProcessor",
    # Document renderers
    "DocxRenderer",
    "DocxStyle",
    "MarkdownRenderer",
    # Tabular data renderers
    "CsvRenderer",
    "ExcelRenderer",
    # Presentation renderer
    "PptxRenderer",
    "SlideType",
    "SlideContent",
]
