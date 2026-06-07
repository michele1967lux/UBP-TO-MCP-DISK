"""
Base Renderer Interface for Multi-Format Artifact Engine (v2.5.1 - FEAT-ARTIFACT-002)

Defines the abstract interface and common utilities for all format renderers.

Architecture:
- FormatRenderer: Abstract base class for all renderers
- OutputFormat: Enum of supported output formats
- RenderResult: Dataclass for render operation results
- RenderContext: Dataclass for render operation context

Usage:
    class MyRenderer(FormatRenderer):
        @property
        def output_format(self) -> OutputFormat:
            return OutputFormat.DOCX

        def render(self, content: str, context: RenderContext) -> RenderResult:
            # Implementation
            pass
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Supported output formats for artifact rendering."""

    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    PPTX = "pptx"
    MD = "md"
    PDF = "pdf"

    @property
    def extension(self) -> str:
        """Get file extension with dot."""
        return f".{self.value}"

    @property
    def mime_type(self) -> str:
        """Get MIME type for format."""
        mime_types = {
            OutputFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            OutputFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            OutputFormat.CSV: "text/csv",
            OutputFormat.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            OutputFormat.MD: "text/markdown",
            OutputFormat.PDF: "application/pdf",
        }
        return mime_types.get(self, "application/octet-stream")


class WorkerMode(str, Enum):
    """Worker modes for content processing."""

    FREE_TEXT = "free_text"           # Standard LLM prose generation
    JSON_EXTRACTION = "json_extraction"  # Extract structured data as JSON
    SLIDE_CHUNKING = "slide_chunking"    # Generate per-slide content
    TEMPLATE_FILL = "template_fill"      # Fill template placeholders


class PostProcessType(str, Enum):
    """Post-processing operations available."""

    NONE = "none"
    APPLY_PRICE_LIST = "apply_price_list"
    CALCULATE_TOTALS = "calculate_totals"
    FORMAT_CURRENCY = "format_currency"


@dataclass
class RenderContext:
    """
    Context for render operations.

    Provides all necessary information for a renderer to produce output.
    """

    # Required fields
    title: str
    content: str

    # Optional metadata
    author: str = "UBP Enterprise"
    subject: str = ""
    language: str = "auto"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Format-specific settings (from formats.yaml)
    format_settings: Dict[str, Any] = field(default_factory=dict)

    # Blueprint configuration
    blueprint_id: str = ""
    worker_mode: WorkerMode = WorkerMode.FREE_TEXT
    post_process: PostProcessType = PostProcessType.NONE

    # Extracted data (for data_extraction types)
    extracted_data: Optional[Union[List[Dict], Dict]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Session information
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize context."""
        if isinstance(self.worker_mode, str):
            self.worker_mode = WorkerMode(self.worker_mode)
        if isinstance(self.post_process, str):
            self.post_process = PostProcessType(self.post_process)


@dataclass
class RenderResult:
    """
    Result of a render operation.

    Contains the rendered content and metadata about the operation.
    """

    # Rendered content as bytes
    content: bytes

    # Output format
    format: OutputFormat

    # Generated filename
    filename: str

    # Content checksum for integrity verification
    checksum: str = ""

    # Size in bytes
    size_bytes: int = 0

    # Success flag
    success: bool = True

    # Error message if failed
    error: Optional[str] = None

    # Render duration in milliseconds
    render_time_ms: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields."""
        if self.content and not self.size_bytes:
            self.size_bytes = len(self.content)
        if self.content and not self.checksum:
            self.checksum = hashlib.sha256(self.content).hexdigest()[:16]

    @classmethod
    def failure(cls, format: OutputFormat, error: str, filename: str = "") -> "RenderResult":
        """Create a failure result."""
        return cls(
            content=b"",
            format=format,
            filename=filename,
            success=False,
            error=error,
        )


class FormatRenderer(ABC):
    """
    Abstract base class for all format renderers.

    Subclasses must implement:
    - output_format: Property returning the OutputFormat this renderer produces
    - render: Method to perform the actual rendering

    Optionally override:
    - validate_content: Pre-render content validation
    - post_process: Post-render transformations
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """
        Initialize renderer with optional settings.

        Args:
            settings: Format-specific configuration from ArtifactSettings
        """
        self._settings = settings or {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info(f"[RENDERER] {self.__class__.__name__} initialized")

    @property
    @abstractmethod
    def output_format(self) -> OutputFormat:
        """Return the output format this renderer produces."""
        pass

    @property
    def extension(self) -> str:
        """Get file extension for this renderer's output."""
        return self.output_format.extension

    @property
    def mime_type(self) -> str:
        """Get MIME type for this renderer's output."""
        return self.output_format.mime_type

    @abstractmethod
    def render(self, context: RenderContext) -> RenderResult:
        """
        Render content to the target format.

        Args:
            context: RenderContext with content and settings

        Returns:
            RenderResult with rendered bytes and metadata
        """
        pass

    def validate_content(self, content: str) -> bool:
        """
        Validate content before rendering.

        Override in subclasses for format-specific validation.

        Args:
            content: Content to validate

        Returns:
            True if content is valid for this renderer
        """
        if not content or not content.strip():
            self._logger.warning("[RENDERER] Empty content provided")
            return False
        return True

    def generate_filename(
        self,
        title: str,
        blueprint_id: str = "",
        timestamp: bool = True,
    ) -> str:
        """
        Generate a filename for the rendered artifact.

        Args:
            title: Document title
            blueprint_id: Blueprint identifier (optional)
            timestamp: Include timestamp in filename

        Returns:
            Generated filename with extension
        """
        import re

        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[\s]+', '_', safe_title)[:50]

        parts = []
        if blueprint_id:
            parts.append(blueprint_id)
        parts.append(safe_title)
        if timestamp:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            parts.append(ts)

        filename = "_".join(parts) + self.extension
        return filename

    def _get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value with fallback to default."""
        return self._settings.get(key, default)

    def _log_render_start(self, context: RenderContext) -> None:
        """Log render operation start."""
        self._logger.info(
            f"[RENDERER] Starting {self.output_format.value} render: "
            f"title='{context.title}', content_len={len(context.content)}"
        )

    def _log_render_complete(self, result: RenderResult, duration_ms: int) -> None:
        """Log render operation completion."""
        if result.success:
            self._logger.info(
                f"[RENDERER] Render complete: {result.filename} "
                f"({result.size_bytes} bytes, {duration_ms}ms)"
            )
        else:
            self._logger.error(
                f"[RENDERER] Render failed: {result.error}"
            )


class DataExtractor:
    """
    Utility class for extracting structured data from content.

    Used by data-extraction type renderers (XLSX, CSV) to parse
    JSON-formatted content from worker responses.
    """

    @staticmethod
    def extract_json(content: str) -> Optional[Union[List, Dict]]:
        """
        Extract JSON data from content.

        Handles common cases:
        - Pure JSON string
        - JSON wrapped in markdown code blocks
        - JSON with surrounding text

        Args:
            content: Content potentially containing JSON

        Returns:
            Parsed JSON data or None if extraction fails
        """
        import json
        import re

        if not content:
            return None

        # Try direct JSON parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, content)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try finding JSON array or object in text
        array_pattern = r'\[\s*\{[\s\S]*\}\s*\]'
        object_pattern = r'\{[\s\S]*\}'

        for pattern in [array_pattern, object_pattern]:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        logger.warning("[EXTRACTOR] Failed to extract JSON from content")
        return None

    @staticmethod
    def validate_columns(
        data: List[Dict],
        required_columns: List[str],
    ) -> bool:
        """
        Validate that extracted data has required columns.

        Args:
            data: List of row dictionaries
            required_columns: Column names that must be present

        Returns:
            True if all required columns exist in all rows
        """
        if not data:
            return False

        for row in data:
            if not isinstance(row, dict):
                return False
            for col in required_columns:
                if col not in row:
                    return False

        return True


class TableDataProcessor:
    """
    Utility class for processing tabular data before rendering.

    Handles:
    - Column normalization
    - Type conversion
    - Formula application
    - Totals calculation
    """

    @staticmethod
    def normalize_columns(
        data: List[Dict],
        column_definitions: List[Dict],
    ) -> List[Dict]:
        """
        Normalize data columns according to definitions.

        Args:
            data: Raw extracted data
            column_definitions: Column definitions from blueprint

        Returns:
            Normalized data with consistent column types
        """
        if not data or not column_definitions:
            return data

        normalized = []
        for row in data:
            norm_row = {}
            for col_def in column_definitions:
                source = col_def.get("source")
                name = col_def.get("name")
                col_type = col_def.get("type", "string")

                if source and source in row:
                    value = row[source]
                elif name and name in row:
                    value = row[name]
                else:
                    value = None

                # Type conversion
                if col_type == "number" and value is not None:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 0.0
                elif col_type == "currency" and value is not None:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 0.0
                elif col_type == "string":
                    value = str(value) if value is not None else ""

                norm_row[name] = value
            normalized.append(norm_row)

        return normalized

    @staticmethod
    def calculate_totals(
        data: List[Dict],
        numeric_columns: List[str],
    ) -> Dict[str, float]:
        """
        Calculate column totals for numeric columns.

        Args:
            data: Normalized data rows
            numeric_columns: Columns to sum

        Returns:
            Dictionary of column name to total
        """
        totals = {col: 0.0 for col in numeric_columns}

        for row in data:
            for col in numeric_columns:
                if col in row and isinstance(row[col], (int, float)):
                    totals[col] += row[col]

        return totals
