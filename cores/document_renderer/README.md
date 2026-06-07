# document_renderer

**Multi-Format Document Rendering Module**

Version: 1.0.0 | Architecture: 3-file-pattern | Pipeline-native

---

## Overview

The `document_renderer` module provides professional document rendering with:

- **PDF Native**: ReportLab-based PDF generation with full control
- **DOCX**: python-docx with custom templates and styles
- **PPTX**: Presentation generation with charts and speaker notes
- **Charts**: matplotlib-powered chart generation
- **Tables**: Professional table formatting with multiple styles

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Format** | PDF, DOCX, PPTX, Markdown, HTML |
| **Custom Templates** | Load and apply custom document templates |
| **Charts** | Bar, line, pie, scatter, heatmap, and more |
| **Tables** | Professional styling with totals and formatting |
| **Cover Pages** | Automatic cover page generation |
| **TOC** | Table of contents for documents |
| **Headers/Footers** | Page numbers and custom text |
| **Parallel Rendering** | Render multiple formats simultaneously |

## Installation

```bash
pip install reportlab>=4.0.0      # For PDF
pip install python-docx>=0.8.11   # For DOCX
pip install python-pptx>=0.6.21   # For PPTX
pip install matplotlib>=3.7.0     # For Charts
pip install Pillow>=10.0.0        # For Images
```

## Quick Start

```python
from document_renderer import create_module
from pathlib import Path

# Create and initialize
renderer = create_module(Path("./document_renderer"))
await renderer.initialize()

# Render a document
result = await renderer.render_pdf(
    content={
        "title": "Quarterly Report",
        "author": "Analysis Team",
        "sections": [
            {
                "title": "Executive Summary",
                "content": "Key findings from Q3...",
                "level": 1
            },
            {
                "title": "Financial Analysis",
                "content": "Revenue grew by 15%...",
                "tables": [{
                    "headers": ["Metric", "Q2", "Q3", "Change"],
                    "rows": [
                        ["Revenue", "€1.2M", "€1.4M", "+16%"],
                        ["Costs", "€800K", "€850K", "+6%"]
                    ],
                    "totals_row": True,
                    "totals_columns": [1, 2]
                }],
                "charts": [{
                    "chart_type": "bar",
                    "title": "Quarterly Revenue",
                    "labels": ["Q1", "Q2", "Q3", "Q4"],
                    "datasets": [{"label": "Revenue", "data": [1.0, 1.2, 1.4, 1.5]}]
                }]
            }
        ]
    },
    options={
        "include_toc": True,
        "include_cover": True
    }
)

# Save or use content
with open("report.pdf", "wb") as f:
    f.write(result["content"])
```

## Operations

### Document Rendering

#### `render_pdf`

Native PDF rendering with full control.

```python
result = await renderer.render_pdf(
    content=document_content,
    options={
        "page_size": "A4",
        "orientation": "portrait",
        "include_toc": True,
        "include_cover": True,
        "include_page_numbers": True,
        "font_family": "Helvetica",
        "font_size": 11,
        "watermark_text": "CONFIDENTIAL",
        "watermark_opacity": 0.1
    },
    output_path="/path/to/output.pdf"
)
```

#### `render_docx`

DOCX with custom templates.

```python
result = await renderer.render_docx(
    content=document_content,
    options={
        "font_family": "Calibri",
        "font_size": 11,
        "line_spacing": 1.15,
        "header_text": "Company Report",
        "footer_text": "Confidential"
    },
    template_id="corporate_template",
    output_path="/path/to/output.docx"
)
```

#### `render_pptx`

Presentation generation.

```python
result = await renderer.render_pptx(
    content={
        "title": "Q3 Results",
        "author": "Finance Team",
        "slides": [
            {
                "title": "Revenue Growth",
                "content": "• 15% increase YoY\n• New markets opened",
                "layout": "content"
            },
            {
                "title": "Financial Overview",
                "chart": {
                    "chart_type": "bar",
                    "title": "Revenue by Region",
                    "labels": ["EMEA", "APAC", "Americas"],
                    "datasets": [{"label": "Q3", "data": [500, 300, 400]}]
                },
                "layout": "chart",
                "notes": "Discuss regional strategy..."
            }
        ]
    },
    options={
        "include_title_slide": True,
        "include_speaker_notes": True
    }
)
```

#### `render_multi`

Render to multiple formats in parallel.

```python
result = await renderer.render_multi(
    content=document_content,
    formats=["pdf", "docx", "pptx"],
    options={
        "pdf": {"watermark_text": "DRAFT"},
        "docx": {"template_id": "corporate"},
        "pptx": {"include_summary_slide": True}
    },
    output_dir="/path/to/output/"
)

# Result
{
    "success": True,
    "formats_rendered": 3,
    "results": {
        "pdf": {"success": True, "pages": 12, "file_size": 245000},
        "docx": {"success": True, "pages": 10, "file_size": 180000},
        "pptx": {"success": True, "slides": 8, "file_size": 150000}
    }
}
```

### Chart Generation

#### `render_chart`

```python
result = await renderer.render_chart(
    chart_data={
        "chart_type": "line",
        "title": "Monthly Trend",
        "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
        "datasets": [
            {"label": "2023", "data": [100, 120, 115, 130, 145]},
            {"label": "2024", "data": [110, 135, 140, 155, 170]}
        ],
        "x_label": "Month",
        "y_label": "Value (€K)",
        "show_legend": True,
        "width": 800,
        "height": 500
    },
    output_format="png"
)
```

**Supported Chart Types:**
- `bar`, `bar_horizontal`, `bar_stacked`
- `line`, `area`
- `pie`, `donut`
- `scatter`, `bubble`
- `heatmap`, `radar`

### Table Rendering

#### `render_table`

```python
result = await renderer.render_table(
    table_data={
        "headers": ["Product", "Q1", "Q2", "Q3", "Total"],
        "rows": [
            ["Widget A", "€10,000", "€12,000", "€15,000", "€37,000"],
            ["Widget B", "€8,000", "€9,500", "€11,000", "€28,500"]
        ],
        "style": "professional",
        "totals_row": True,
        "totals_columns": [1, 2, 3, 4],
        "caption": "Quarterly Sales by Product"
    },
    output_format="html"
)
```

**Table Styles:**
- `professional` - Blue header, alternating rows
- `minimal` - Clean, minimal borders
- `bordered` - Full borders
- `striped` - Alternating row colors
- `colorful` - Orange accent color

## Content Structure

### DocumentContent

```python
content = {
    "title": "Document Title",
    "author": "Author Name",
    "language": "it",  # or "en"
    "include_toc": True,
    "include_cover": True,
    "sections": [
        {
            "id": "section_1",
            "title": "Section Title",
            "content": "Section content text...",
            "level": 1,  # Heading level (1-3)
            "tables": [...],
            "charts": [...],
            "images": [...],
            "page_break_before": False
        }
    ],
    "bibliography": "Formatted bibliography text...",
    "style": {
        "primary_color": "#1F4E79",
        "secondary_color": "#4472C4",
        "table_style": "professional"
    }
}
```

### PresentationContent

```python
presentation = {
    "title": "Presentation Title",
    "subtitle": "Subtitle",
    "author": "Presenter Name",
    "slides": [
        {
            "title": "Slide Title",
            "content": "• Bullet point 1\n• Bullet point 2",
            "layout": "content",  # content, chart, table, image
            "notes": "Speaker notes...",
            "chart": {...},  # Optional
            "table": {...},  # Optional
            "image": {...}   # Optional
        }
    ]
}
```

## Pipeline Integration

```yaml
name: document_generation_pipeline
steps:
  - id: render_report
    module: document_renderer
    operation: render_multi
    params:
      formats: ["pdf", "docx"]
    input_from:
      content:
        title: plan.title
        sections: generated_content.sections
        bibliography: bibliography.formatted_text
      options:
        pdf:
          include_toc: true
          watermark_text: ${config.watermark|default:null}
    output_as: rendered_documents
    enabled: true

  - id: render_charts
    module: document_renderer
    operation: render_charts_batch
    input_from:
      charts: generated_content.charts
    output_as: chart_images
    parallel: true
```

## Styling

### Style Configuration

```python
style = {
    "page_size": "A4",           # A4, A3, Letter, Legal
    "orientation": "portrait",    # portrait, landscape
    "primary_color": "#1F4E79",   # Headers, titles
    "secondary_color": "#4472C4", # Subheadings
    "accent_color": "#ED7D31",    # Highlights
    "table_style": "professional",
    "header_text": "Company Name",
    "footer_text": "Confidential",
    "include_page_numbers": True
}
```

### Color Palette

Default professional colors:
- Primary: `#1F4E79` (Dark Blue)
- Secondary: `#4472C4` (Blue)
- Accent: `#ED7D31` (Orange)
- Neutral: `#A5A5A5` (Gray)

## Configuration

Environment variables:

```bash
DOCUMENT_RENDERER__FORMAT=pdf
DOCUMENT_RENDERER__PAGE_SIZE=A4
DOCUMENT_RENDERER__PRIMARY_COLOR=#1F4E79
DOCUMENT_RENDERER__PDF_FONT=Helvetica
DOCUMENT_RENDERER__DOCX_FONT=Calibri
DOCUMENT_RENDERER__CHART_DPI=150
DOCUMENT_RENDERER__TABLE_STYLE=professional
DOCUMENT_RENDERER__TEMPLATES_DIR=/app/templates
```

## Custom Templates

Place templates in the templates directory:

```
templates/
├── docx/
│   ├── corporate.docx
│   └── minimal.docx
├── pptx/
│   ├── professional.pptx
│   └── creative.pptx
└── styles/
    └── custom_style.yaml
```

---

**Module**: document_renderer v1.0.0 | **Architecture**: Pipeline-native | **Status**: Production Ready
