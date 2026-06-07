# content_planner

**Document Structure Planning Module**

Version: 1.0.0 | Architecture: 3-file-pattern | Pipeline-native

---

## Overview

The `content_planner` module provides intelligent document structure planning with:

- **Template-based Planning**: Use predefined templates for common document types
- **Dynamic Planning**: LLM-based structure generation for custom documents
- **Microprompt Generation**: Section-specific prompts for high-quality content
- **Plan Validation**: Comprehensive validation with actionable suggestions
- **Interactive Modification**: Add, remove, modify, and reorder sections

## Features

| Feature | Description |
|---------|-------------|
| **Template Matching** | Automatic template selection based on query analysis |
| **Microprompts** | Section-specific LLM prompts with style, tone, structure guidance |
| **Dependency Management** | Section dependencies for proper execution order |
| **Token Estimation** | Budget tracking and resource estimation |
| **Validation** | Structure, constraint, and dependency validation |
| **Multi-language** | Support for Italian, English, and other languages |

## Installation

```bash
# The module has minimal dependencies
pip install PyYAML>=6.0
```

## Quick Start

```python
from content_planner import create_module
from pathlib import Path

# Create and initialize
planner = create_module(Path("./content_planner"))
await planner.initialize()

# Plan a document
result = await planner.plan_structure(
    query="Create a technical analysis of our new microservices architecture",
    constraints={
        "max_sections": 8,
        "language": "en",
        "formality_level": "technical"
    },
    collections=["architecture_docs", "best_practices"]
)

plan = result["plan"]
print(f"Title: {plan['title']}")
print(f"Sections: {len(plan['sections'])}")
```

## Operations

### Planning

#### `plan_structure`

Generate a complete document plan from a query.

```python
result = await planner.plan_structure(
    query="Computo metrico per ristrutturazione bagno 6mq",
    constraints={
        "max_sections": 10,
        "max_tokens_total": 10000,
        "language": "it",
        "formality_level": "professional",
        "include_citations": True
    },
    template_id=None,  # Auto-match template
    collections=["prezziari", "normative"],
    auto_validate=True
)

# Result
{
    "success": True,
    "plan": {
        "id": "abc123",
        "title": "Computo Metrico - Ristrutturazione Bagno",
        "document_type": "technical",
        "sections": [
            {
                "id": "intestazione",
                "title": "Intestazione",
                "section_type": "introduction",
                "content_type": "prose",
                "microprompt": {...},
                "target_tokens": 250,
                "source_preference": "llm_reasoning"
            },
            ...
        ]
    },
    "validation": {"is_valid": True, "issues": []},
    "estimated_tokens": 3500
}
```

#### `plan_presentation`

Generate a plan specifically for presentations.

```python
result = await planner.plan_presentation(
    query="Quarterly sales review for board meeting",
    max_slides=15,
    style="professional",
    collections=["sales_data"]
)
```

### Section Management

#### `add_section`

```python
result = await planner.add_section(
    plan=existing_plan,
    section={
        "id": "market_analysis",
        "title": "Market Analysis",
        "section_type": "analysis",
        "content_type": "mixed",
        "target_tokens": 800
    },
    position=2  # Insert at position 2
)
```

#### `modify_section`

```python
result = await planner.modify_section(
    plan=existing_plan,
    section_id="findings",
    changes={
        "target_tokens": 1500,
        "content_type": "mixed",
        "interactive_review": True
    }
)
```

#### `reorder_sections`

```python
result = await planner.reorder_sections(
    plan=existing_plan,
    new_order=["intro", "analysis", "findings", "recommendations", "conclusion"]
)
```

### Templates

#### `list_templates`

```python
result = await planner.list_templates(
    category="technical",
    document_type="report"
)
# Returns: {"templates": [...], "count": 5, "categories": ["research", "technical", ...]}
```

#### `match_template`

```python
result = await planner.match_template(
    query="Preventivo per lavori di ristrutturazione",
    min_confidence=0.4
)
# Returns: {"matched": True, "match": {"template_id": "computo_metrico", "confidence": 0.85}}
```

### Microprompts

#### `generate_microprompt`

```python
result = await planner.generate_microprompt(
    section={
        "id": "findings",
        "title": "Key Findings",
        "section_type": "findings",
        "content_type": "mixed"
    },
    context={
        "document_title": "Market Research Report",
        "language": "en",
        "formality": "professional"
    }
)

# Result includes:
# - microprompt: Full microprompt object
# - system_prompt: Ready-to-use system prompt for LLM
# - generation_prompt: User prompt template
```

### Validation & Estimation

#### `validate_plan`

```python
result = await planner.validate_plan(plan)
# Returns validation issues with severity and suggestions
```

#### `estimate_resources`

```python
result = await planner.estimate_resources(plan)
# Returns:
{
    "estimated_time_minutes": 15,
    "estimated_api_calls": 12,
    "parallel_batches": 3,
    "sections_requiring_research": 4,
    "sections_llm_only": 2
}
```

## Built-in Templates

| Template ID | Name | Category | Sections |
|-------------|------|----------|----------|
| `research_report` | Research Report | research | 5 (summary, intro, methodology, findings, conclusion) |
| `technical_analysis` | Technical Analysis | technical | 4 (overview, requirements, details, recommendations) |
| `computo_metrico` | Computo Metrico | construction | 4 (intestazione, descrizione, voci, riepilogo) |
| `executive_brief` | Executive Brief | executive | 3 (situation, analysis, recommendations) |

## Custom Templates

Create custom templates in YAML:

```yaml
# templates/my_template.yaml
id: my_custom_template
name: "My Custom Report"
description: "Custom report for specific needs"
category: custom
document_type: report
keywords:
  - custom
  - specific
selection_patterns:
  - "(?i)custom|specific|special"

default_formality: professional

sections:
  - id: overview
    title_template: "Overview of {topic}"
    section_type: introduction
    content_type: prose
    microprompt_template: |
      Write an overview that:
      - Introduces the topic
      - Sets context
      - Outlines scope
    default_tokens: 500
    required: true
    order: 1

  - id: details
    title_template: "Detailed Analysis"
    section_type: analysis
    content_type: mixed
    source_preference: rag_first
    microprompt_template: |
      Provide detailed analysis with:
      - Data and evidence
      - Tables where appropriate
      - Clear conclusions
    default_tokens: 1200
    required: true
    order: 2
```

## Pipeline Integration

Use in pipeline templates:

```yaml
name: document_generation_pipeline
steps:
  - id: plan
    module: content_planner
    operation: plan_structure
    params:
      auto_validate: true
    input_from:
      query: inputs.query
      constraints: inputs.constraints
      collections: inputs.collections
    output_as: plan
    enabled: true

  # Optional: Interactive plan approval
  - id: plan_approval
    module: _builtin
    operation: checkpoint
    params:
      checkpoint_type: approval
      allow_modifications: true
    input_from:
      data: plan
    output_as: approved_plan
    enabled: ${config.interactive|default:false}

  - id: research
    module: swarm_researcher
    operation: research_parallel
    input_from:
      queries: approved_plan.sections[*].suggested_queries
    output_as: research_data
```

## Data Classes

### StructuredPlan

```python
@dataclass
class StructuredPlan:
    id: str
    title: str
    description: str
    document_type: DocumentType
    sections: List[SectionPlan]
    constraints: PlanConstraints
    metadata: PlanMetadata
    estimated_tokens: int
    collections: List[str]
    language: str
```

### SectionPlan

```python
@dataclass
class SectionPlan:
    id: str
    title: str
    description: str
    order: int
    section_type: SectionType
    content_type: ContentType
    microprompt: Optional[Microprompt]
    source_preference: SourcePreference
    suggested_queries: List[str]
    depends_on: List[str]
    min_tokens: int
    max_tokens: int
    target_tokens: int
    required: bool
    interactive_review: bool
    enabled: bool
```

### Microprompt

```python
@dataclass
class Microprompt:
    section_id: str
    section_type: SectionType
    system_context: str
    generation_prompt: str
    writing_style: WritingStyle
    tone: Tone
    structure_elements: List[str]
    include_citations: bool
    min_tokens: int
    max_tokens: int
    quality_criteria: List[str]
```

## Configuration

Environment variables:

```bash
CONTENT_PLANNER__LANGUAGE=it
CONTENT_PLANNER__FORMALITY=professional
CONTENT_PLANNER__MODEL=grok/grok-3-fast
CONTENT_PLANNER__MAX_SECTIONS=15
CONTENT_PLANNER__MAX_TOKENS_TOTAL=15000
CONTENT_PLANNER__TEMPLATES_DIR=/app/templates
CONTENT_PLANNER__AUTO_MATCH=true
```

## Error Handling

```python
result = await planner.plan_structure(query="...")

if not result["success"]:
    print(f"Error: {result['error']}")
    return

if result["validation"] and not result["validation"]["is_valid"]:
    for issue in result["validation"]["issues"]:
        print(f"[{issue['severity']}] {issue['message']}")
        if issue.get("suggestion"):
            print(f"  Suggestion: {issue['suggestion']}")
```

## Best Practices

1. **Use Templates When Possible**: Templates provide consistent, well-tested structures
2. **Set Appropriate Constraints**: Define token budgets and section limits upfront
3. **Validate Plans**: Always validate before execution
4. **Use Dependencies**: Define section dependencies for proper research order
5. **Leverage Microprompts**: Let microprompts guide content generation for consistency

---

**Module**: content_planner v1.0.0 | **Architecture**: Pipeline-native | **Status**: Production Ready
