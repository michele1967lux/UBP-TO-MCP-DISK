# citation_manager

**Citation Management and Bibliography Generation Module**

Version: 1.0.0 | Architecture: 3-file-pattern | Pipeline-native

---

## Overview

The `citation_manager` module provides professional citation management with:

- **Multi-Style Formatting**: APA, MLA, Chicago, IEEE, Harvard, Vancouver
- **Persistent Storage**: JSON-based with automatic deduplication
- **Cross-Document Tracking**: Track citations across sections and documents
- **Validation**: Field validation and duplicate detection
- **Export**: BibTeX and JSON export

## Features

| Feature | Description |
|---------|-------------|
| **13 Citation Styles** | APA, MLA, Chicago, IEEE, Harvard, Vancouver, and more |
| **Deduplication** | Automatic by DOI, URL, or title similarity |
| **Validation** | Field validation with suggestions |
| **Inline Formatting** | Author-date and numeric formats |
| **Cross-Document** | Track citations across sections |
| **BibTeX Export** | Standard academic format |
| **RAG Integration** | Add citations from retrieved documents |

## Quick Start

```python
from citation_manager import create_module
from pathlib import Path

# Create and initialize
manager = create_module(Path("./citation_manager"))
await manager.initialize(default_style="apa")

# Add a citation
result = await manager.add_citation(
    title="Deep Learning for Natural Language Processing",
    authors=[
        {"last_name": "Smith", "first_name": "John"},
        {"last_name": "Doe", "first_name": "Jane"}
    ],
    year="2024",
    source_type="journal",
    journal="Nature Machine Intelligence",
    volume="6",
    pages="123-145",
    doi="10.1038/s42256-024-00123-4"
)

print(f"Added: {result['citation_id']}")

# Format citation
formatted = await manager.format_citation(
    citation_id=result["citation_id"],
    style="apa"
)
print(formatted["formatted"])
# Output: Smith, J., & Doe, J. (2024). Deep Learning for Natural Language Processing. *Nature Machine Intelligence*, *6*, 123-145. https://doi.org/10.1038/s42256-024-00123-4
```

## Operations

### Adding Citations

```python
# Manual citation
result = await manager.add_citation(
    title="Machine Learning",
    authors=[{"last_name": "Smith", "first_name": "John"}],
    year="2024",
    source_type="book",
    publisher="MIT Press",
    isbn="978-0-262-12345-6"
)

# From retrieved document (RAG integration)
result = await manager.add_from_document(
    document={
        "id": "doc_123",
        "content": "Relevant text...",
        "metadata": {
            "title": "Paper Title",
            "authors": "Smith, J. and Doe, J.",
            "year": "2024",
            "doi": "10.1234/example"
        },
        "score": 0.95
    },
    section_id="intro"
)
```

### Formatting Citations

```python
# Single citation
formatted = await manager.format_citation(
    citation_id="abc123",
    style="ieee"
)

# Inline reference
inline = await manager.format_inline(
    citation_id="abc123",
    number=1,
    page="42",
    style="apa"
)
# Output: (Smith, 2024, p. 42)

# For numeric styles
inline = await manager.format_inline(
    citation_id="abc123",
    number=5,
    style="ieee"
)
# Output: [5]
```

### Generating Bibliography

```python
# All citations
bib = await manager.generate_bibliography(
    style="apa",
    title="References",
    numbered=True,
    sort_by="author"
)

# For specific section
bib = await manager.generate_bibliography(
    section_id="methodology",
    style="chicago"
)

# Specific citations
bib = await manager.generate_bibliography(
    citation_ids=["id1", "id2", "id3"],
    style="ieee"
)
```

### Validation

```python
# Validate single citation
validation = await manager.validate_citation(citation_id="abc123")

# Validate all
result = await manager.validate_all()
print(f"Valid: {result['valid']}/{result['total']}")

# Find duplicates
duplicates = await manager.find_duplicates()
for dup in duplicates["duplicates"]:
    print(f"Duplicate: {dup['id1']} and {dup['id2']} ({dup['reason']})")
```

### Export

```python
# BibTeX export
bibtex = await manager.export_bibtex()
with open("references.bib", "w") as f:
    f.write(bibtex["content"])

# JSON export
json_data = await manager.export_json()
```

## Citation Styles

| Style | Format | Example Inline |
|-------|--------|----------------|
| APA 7 | Author-Date | (Smith, 2024) |
| MLA 9 | Author-Page | (Smith 42) |
| Chicago | Author-Date or Notes | (Smith 2024) |
| IEEE | Numeric | [1] |
| Harvard | Author-Date | (Smith 2024) |
| Vancouver | Numeric | (1) |

### Style Examples

**APA 7:**
```
Smith, J., & Doe, J. (2024). Title of article. *Journal Name*, *12*(3), 45-67. https://doi.org/10.1234/example
```

**MLA 9:**
```
Smith, John, and Jane Doe. "Title of Article." *Journal Name*, vol. 12, no. 3, 2024, pp. 45-67.
```

**IEEE:**
```
J. Smith and J. Doe, "Title of article," *Journal Name*, vol. 12, no. 3, pp. 45-67, 2024.
```

## Source Types

- `book` - Books and ebooks
- `journal` - Journal articles
- `article` - General articles
- `conference` - Conference papers
- `website` - Web pages
- `report` - Technical reports
- `thesis` - Dissertations
- `rag_document` - RAG retrieved documents
- `other` - Other sources

## Pipeline Integration

```yaml
name: report_with_citations
steps:
  - id: research
    module: swarm_researcher
    operation: research_parallel
    output_as: research_data

  - id: track_citations
    module: citation_manager
    operation: add_from_document
    foreach: research_data.documents
    input_from:
      document: ${item}
      section_id: ${item.section_id}
    output_as: citations

  - id: generate_bibliography
    module: citation_manager
    operation: generate_bibliography
    params:
      style: apa
      numbered: true
    output_as: bibliography

  - id: render
    module: document_renderer
    operation: render_pdf
    input_from:
      content.bibliography: bibliography.bibliography
```

## Configuration

Environment variables:

```bash
CITATION_MANAGER__STYLE=apa
CITATION_MANAGER__NUMBERED=true
CITATION_MANAGER__SORT=author
CITATION_MANAGER__STRICT=false
CITATION_MANAGER__DEDUP=true
CITATION_MANAGER__SIMILARITY=0.85
CITATION_MANAGER__PERSIST_PATH=/app/data/citations.json
```

## Validation Rules

| Field | Rule | Severity |
|-------|------|----------|
| title | Required | Error |
| authors | Required for most types | Warning |
| year | Recommended | Warning |
| doi | Must match pattern 10.xxxx/ | Error |
| url | Must start with http(s):// | Warning |
| journal | Required for journal type | Warning |

---

**Module**: citation_manager v1.0.0 | **Architecture**: Pipeline-native | **Status**: Production Ready
