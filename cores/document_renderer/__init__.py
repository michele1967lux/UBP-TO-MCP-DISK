"""
document_renderer/providers/__init__.py

Data classes and enums for document rendering.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


# ============================================================================
# Enums
# ============================================================================


class OutputFormat(str, Enum):
    """Supported output formats."""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    MARKDOWN = "md"
    HTML = "html"


class PageSize(str, Enum):
    """Standard page sizes."""
    A4 = "A4"
    A3 = "A3"
    LETTER = "Letter"
    LEGAL = "Legal"
    TABLOID = "Tabloid"


class Orientation(str, Enum):
    """Page orientation."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    BAR_HORIZONTAL = "bar_horizontal"
    BAR_STACKED = "bar_stacked"
    LINE = "line"
    LINE_AREA = "area"
    PIE = "pie"
    DONUT = "donut"
    SCATTER = "scatter"
    BUBBLE = "bubble"
    RADAR = "radar"
    WATERFALL = "waterfall"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"


class TableStyle(str, Enum):
    """Table styling options."""
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    BORDERED = "bordered"
    STRIPED = "striped"
    COLORFUL = "colorful"


class ImagePosition(str, Enum):
    """Image positioning options."""
    INLINE = "inline"
    FLOAT_LEFT = "float_left"
    FLOAT_RIGHT = "float_right"
    CENTER = "center"
    FULL_WIDTH = "full_width"


# ============================================================================
# Style Configuration
# ============================================================================


@dataclass
class Margins:
    """Page margins in cm."""
    top: float = 2.5
    bottom: float = 2.5
    left: float = 2.5
    right: float = 2.5


@dataclass
class FontConfig:
    """Font configuration."""
    family: str = "Arial"
    size: int = 11
    color: str = "#000000"
    bold: bool = False
    italic: bool = False


@dataclass
class StyleConfig:
    """Complete style configuration for documents."""
    # Page setup
    page_size: PageSize = PageSize.A4
    orientation: Orientation = Orientation.PORTRAIT
    margins: Margins = field(default_factory=Margins)
    
    # Typography
    font_body: FontConfig = field(default_factory=lambda: FontConfig(family="Arial", size=11))
    font_heading1: FontConfig = field(default_factory=lambda: FontConfig(family="Arial", size=16, bold=True))
    font_heading2: FontConfig = field(default_factory=lambda: FontConfig(family="Arial", size=14, bold=True))
    font_heading3: FontConfig = field(default_factory=lambda: FontConfig(family="Arial", size=12, bold=True))
    line_spacing: float = 1.15
    
    # Colors
    primary_color: str = "#1F4E79"
    secondary_color: str = "#4472C4"
    accent_color: str = "#ED7D31"
    
    # Tables
    table_style: TableStyle = TableStyle.PROFESSIONAL
    table_header_bg: str = "#1F4E79"
    table_header_text: str = "#FFFFFF"
    table_alternate_row: str = "#F2F2F2"
    
    # Headers/Footers
    header_text: Optional[str] = None
    footer_text: Optional[str] = None
    include_page_numbers: bool = True
    page_number_format: str = "Page {page} of {total}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_size": self.page_size.value,
            "orientation": self.orientation.value,
            "font_body": self.font_body.family,
            "primary_color": self.primary_color,
            "table_style": self.table_style.value,
        }


# ============================================================================
# Document Content
# ============================================================================


@dataclass
class TableCell:
    """Single table cell."""
    content: str
    colspan: int = 1
    rowspan: int = 1
    bold: bool = False
    align: str = "left"  # left, center, right
    bg_color: Optional[str] = None
    text_color: Optional[str] = None


@dataclass
class TableData:
    """Table data structure."""
    headers: List[str]
    rows: List[List[Union[str, TableCell]]]
    caption: Optional[str] = None
    style: TableStyle = TableStyle.PROFESSIONAL
    column_widths: Optional[List[float]] = None  # Percentages
    
    # Advanced features
    merge_cells: List[Dict[str, int]] = field(default_factory=list)  # [{row, col, rowspan, colspan}]
    totals_row: bool = False
    totals_columns: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "headers": self.headers,
            "rows_count": len(self.rows),
            "style": self.style.value,
            "has_totals": self.totals_row,
        }


@dataclass
class ChartData:
    """Data for chart generation."""
    chart_type: ChartType
    title: str
    
    # Data
    labels: List[str] = field(default_factory=list)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    # datasets format: [{"label": "Series 1", "data": [1, 2, 3], "color": "#..."}, ...]
    
    # Axes
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    
    # Styling
    colors: Optional[List[str]] = None
    show_legend: bool = True
    legend_position: str = "right"  # top, bottom, left, right
    show_grid: bool = True
    show_values: bool = False
    
    # Size
    width: int = 600
    height: int = 400
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "labels_count": len(self.labels),
            "datasets_count": len(self.datasets),
        }


@dataclass
class ImageData:
    """Image data for embedding."""
    source: Union[str, bytes, Path]  # URL, bytes, or file path
    width: Optional[int] = None  # Pixels or None for auto
    height: Optional[int] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    position: ImagePosition = ImagePosition.CENTER
    
    # Future: AI generation placeholder
    ai_prompt: Optional[str] = None  # For future AI image generation


@dataclass
class SectionContent:
    """Content for a document section."""
    id: str
    title: str
    content: str  # Markdown or HTML
    
    # Embedded elements
    tables: List[TableData] = field(default_factory=list)
    charts: List[ChartData] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    
    # Formatting
    level: int = 1  # Heading level
    page_break_before: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content_length": len(self.content),
            "tables_count": len(self.tables),
            "charts_count": len(self.charts),
            "images_count": len(self.images),
        }


@dataclass
class DocumentContent:
    """Complete document content for rendering."""
    title: str
    sections: List[SectionContent]
    
    # Metadata
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    language: str = "it"
    
    # Structure
    include_toc: bool = True
    include_cover: bool = True
    
    # Bibliography
    bibliography: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Styling
    style: StyleConfig = field(default_factory=StyleConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "sections_count": len(self.sections),
            "author": self.author,
            "language": self.language,
            "include_toc": self.include_toc,
            "has_bibliography": self.bibliography is not None,
        }


# ============================================================================
# Slide Content (for PPTX)
# ============================================================================


@dataclass
class SlideContent:
    """Content for a single slide."""
    title: str
    content: str  # Main content (bullet points or text)
    
    # Optional elements
    subtitle: Optional[str] = None
    notes: Optional[str] = None  # Speaker notes
    chart: Optional[ChartData] = None
    table: Optional[TableData] = None
    image: Optional[ImageData] = None
    
    # Layout
    layout: str = "content"  # title, content, two_column, image, chart, table
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "layout": self.layout,
            "has_chart": self.chart is not None,
            "has_table": self.table is not None,
            "has_image": self.image is not None,
        }


@dataclass
class PresentationContent:
    """Complete presentation content."""
    title: str
    slides: List[SlideContent]
    
    # Metadata
    author: Optional[str] = None
    subtitle: Optional[str] = None
    
    # Structure
    include_title_slide: bool = True
    include_toc_slide: bool = False
    include_summary_slide: bool = False
    
    # Styling
    style: StyleConfig = field(default_factory=StyleConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "slides_count": len(self.slides),
            "author": self.author,
        }


# ============================================================================
# Render Options
# ============================================================================


@dataclass
class PdfOptions:
    """Options for PDF rendering."""
    # Page setup
    page_size: PageSize = PageSize.A4
    orientation: Orientation = Orientation.PORTRAIT
    margins: Margins = field(default_factory=Margins)
    
    # Features
    include_toc: bool = True
    include_cover: bool = True
    include_page_numbers: bool = True
    include_bookmarks: bool = True
    
    # Security
    encrypt: bool = False
    password: Optional[str] = None
    allow_printing: bool = True
    allow_copying: bool = True
    
    # Watermark
    watermark_text: Optional[str] = None
    watermark_opacity: float = 0.1
    
    # Typography
    font_family: str = "Helvetica"
    font_size: int = 11


@dataclass
class DocxOptions:
    """Options for DOCX rendering."""
    # Template
    template_path: Optional[str] = None
    
    # Page setup
    page_size: PageSize = PageSize.A4
    orientation: Orientation = Orientation.PORTRAIT
    margins: Margins = field(default_factory=Margins)
    
    # Features
    include_toc: bool = True
    include_cover: bool = True
    include_page_numbers: bool = True
    include_bibliography: bool = True
    
    # Typography
    font_family: str = "Calibri"
    font_size: int = 11
    line_spacing: float = 1.15
    
    # Headers/Footers
    header_text: Optional[str] = None
    footer_text: Optional[str] = None


@dataclass
class PptxOptions:
    """Options for PPTX rendering."""
    # Template
    template_path: Optional[str] = None
    
    # Slide setup
    width_inches: float = 13.333  # 16:9
    height_inches: float = 7.5
    
    # Features
    include_title_slide: bool = True
    include_toc_slide: bool = False
    include_summary_slide: bool = False
    include_speaker_notes: bool = True
    
    # Styling
    theme: str = "professional"  # professional, minimal, colorful
    font_title: str = "Calibri"
    font_body: str = "Calibri"


@dataclass
class XlsxOptions:
    """Options for XLSX rendering."""
    # Sheet setup
    sheet_name: str = "Data"
    freeze_header: bool = True
    auto_filter: bool = True
    
    # Styling
    header_style: str = "professional"
    alternate_rows: bool = True
    
    # Formatting
    number_format: Optional[str] = None
    date_format: str = "DD/MM/YYYY"


# ============================================================================
# Render Results
# ============================================================================


@dataclass
class RenderResult:
    """Result of a render operation."""
    success: bool
    format: OutputFormat
    content: Optional[bytes] = None
    
    # Metadata
    pages: int = 0
    file_size: int = 0
    render_time_ms: int = 0
    
    # Error info
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "format": self.format.value,
            "pages": self.pages,
            "file_size": self.file_size,
            "render_time_ms": self.render_time_ms,
            "error": self.error,
        }


@dataclass
class ChartResult:
    """Result of chart generation."""
    success: bool
    chart_type: ChartType
    image_bytes: Optional[bytes] = None
    image_format: str = "png"  # png, svg
    width: int = 0
    height: int = 0
    error: Optional[str] = None


@dataclass
class MultiRenderResult:
    """Result of multi-format rendering."""
    success: bool
    results: Dict[str, RenderResult] = field(default_factory=dict)
    total_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "formats": list(self.results.keys()),
            "total_time_ms": self.total_time_ms,
        }


# ============================================================================
# Template Info
# ============================================================================


@dataclass
class TemplateInfo:
    """Information about a document template."""
    id: str
    name: str
    description: str
    format: OutputFormat
    path: str
    
    # Metadata
    created_at: Optional[datetime] = None
    preview_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "format": self.format.value,
        }


# Export all
__all__ = [
    # Enums
    "OutputFormat",
    "PageSize",
    "Orientation",
    "ChartType",
    "TableStyle",
    "ImagePosition",
    # Style
    "Margins",
    "FontConfig",
    "StyleConfig",
    # Content
    "TableCell",
    "TableData",
    "ChartData",
    "ImageData",
    "SectionContent",
    "DocumentContent",
    "SlideContent",
    "PresentationContent",
    # Options
    "PdfOptions",
    "DocxOptions",
    "PptxOptions",
    "XlsxOptions",
    # Results
    "RenderResult",
    "ChartResult",
    "MultiRenderResult",
    "TemplateInfo",
]
