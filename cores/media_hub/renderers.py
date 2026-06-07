"""media_hub renderers — Chart (matplotlib), Diagram (mermaid), Fake.

Each renderer implements a simple protocol:
    async def render(spec, data, constraints, output_dir) -> dict
        Returns {"artifacts": [...], "engine": "...", "status": "ok"|"error"}
"""
from __future__ import annotations

import hashlib
import io
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("ubp.media_hub.renderers")


# ══════════════════════════════════════════════════════════
#  Renderer Protocol
# ══════════════════════════════════════════════════════════

class RendererProtocol:
    """Protocol for all media renderers."""

    async def render(
        self,
        spec: Dict[str, Any],
        data: Optional[Dict[str, Any]],
        constraints: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════
#  Matplotlib Chart Renderer
# ══════════════════════════════════════════════════════════

class MatplotlibChartRenderer(RendererProtocol):
    """Renders charts using matplotlib. Produces PNG and/or SVG."""

    SUPPORTED_TYPES = {"bar", "line", "scatter", "pie", "area", "histogram", "box", "heatmap"}

    def __init__(self, default_dpi: int = 150, default_style: str = "seaborn-v0_8"):
        self.default_dpi = default_dpi
        self.default_style = default_style

    async def render(
        self,
        spec: Dict[str, Any],
        data: Optional[Dict[str, Any]],
        constraints: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Render a chart to PNG/SVG."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        chart_type = spec.get("chart_type", "bar")
        if chart_type not in self.SUPPORTED_TYPES:
            return {"status": "error", "error": f"Unsupported chart type: {chart_type}", "artifacts": []}

        if not data or not data.get("columns") or not data.get("rows"):
            return {"status": "error", "error": "No chart data provided", "artifacts": []}

        try:
            # Apply style
            style = spec.get("style", self.default_style)
            available = plt.style.available
            if style in available:
                plt.style.use(style)

            dpi = constraints.get("dpi", self.default_dpi)
            output_format = constraints.get("output_format", "png")
            title = spec.get("title", "")

            columns = data["columns"]
            rows = data["rows"]

            fig, ax = plt.subplots(figsize=(10, 6))

            self._plot(ax, chart_type, columns, rows, spec)

            if title:
                ax.set_title(title, fontsize=14, fontweight="bold")
            if spec.get("x_label"):
                ax.set_xlabel(spec["x_label"])
            if spec.get("y_label"):
                ax.set_ylabel(spec["y_label"])

            plt.tight_layout()

            # Save
            os.makedirs(output_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            artifacts = []

            for fmt in (["png", "svg"] if output_format == "both" else [output_format]):
                filename = f"chart_{chart_type}_{ts}.{fmt}"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches="tight")
                size_bytes = os.path.getsize(filepath)

                # Compute content hash
                with open(filepath, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()[:16]

                w, h = fig.get_size_inches()
                artifacts.append({
                    "uri": filepath,
                    "format": fmt,
                    "size_bytes": size_bytes,
                    "width_px": int(w * dpi),
                    "height_px": int(h * dpi),
                    "content_hash": content_hash,
                })

            plt.close(fig)

            return {
                "status": "ok",
                "engine": "matplotlib",
                "artifacts": artifacts,
            }

        except Exception as e:
            logger.error(f"Chart render error: {e}")
            plt.close("all")
            return {"status": "error", "error": str(e), "artifacts": []}

    def _plot(
        self, ax, chart_type: str, columns: List[str], rows: List[list], spec: Dict[str, Any]
    ):
        """Dispatch to the right plot function."""
        import matplotlib.pyplot as plt

        if chart_type == "bar":
            self._plot_bar(ax, columns, rows, spec)
        elif chart_type == "line":
            self._plot_line(ax, columns, rows, spec)
        elif chart_type == "scatter":
            self._plot_scatter(ax, columns, rows, spec)
        elif chart_type == "pie":
            self._plot_pie(ax, columns, rows, spec)
        elif chart_type == "area":
            self._plot_area(ax, columns, rows, spec)
        elif chart_type == "histogram":
            self._plot_histogram(ax, columns, rows, spec)
        elif chart_type == "box":
            self._plot_box(ax, columns, rows, spec)
        elif chart_type == "heatmap":
            self._plot_heatmap(ax, columns, rows, spec)

    def _plot_bar(self, ax, columns, rows, spec):
        if len(columns) >= 2:
            labels = [str(r[0]) for r in rows]
            values = [r[1] if len(r) > 1 else 0 for r in rows]
            colors = spec.get("colors")
            ax.bar(labels, values, color=colors[0] if colors else None)
        else:
            values = [r[0] for r in rows]
            ax.bar(range(len(values)), values)

    def _plot_line(self, ax, columns, rows, spec):
        if len(columns) >= 2:
            x = [r[0] for r in rows]
            for i in range(1, len(columns)):
                y = [r[i] if len(r) > i else 0 for r in rows]
                ax.plot(x, y, label=columns[i], marker="o")
            if len(columns) > 2:
                ax.legend()
        else:
            values = [r[0] for r in rows]
            ax.plot(values, marker="o")

    def _plot_scatter(self, ax, columns, rows, spec):
        if len(columns) >= 2:
            x = [r[0] for r in rows]
            y = [r[1] if len(r) > 1 else 0 for r in rows]
            ax.scatter(x, y)

    def _plot_pie(self, ax, columns, rows, spec):
        if len(columns) >= 2:
            labels = [str(r[0]) for r in rows]
            values = [r[1] if len(r) > 1 else 0 for r in rows]
            ax.pie(values, labels=labels, autopct="%1.1f%%")
        else:
            values = [r[0] for r in rows]
            ax.pie(values, autopct="%1.1f%%")

    def _plot_area(self, ax, columns, rows, spec):
        if len(columns) >= 2:
            x = [r[0] for r in rows]
            y = [r[1] if len(r) > 1 else 0 for r in rows]
            ax.fill_between(range(len(x)), y, alpha=0.4)
            ax.plot(range(len(x)), y)

    def _plot_histogram(self, ax, columns, rows, spec):
        values = [r[0] for r in rows]
        bins = spec.get("bins", 10)
        ax.hist(values, bins=bins, edgecolor="black")

    def _plot_box(self, ax, columns, rows, spec):
        # Each column becomes a box
        data_cols = []
        for i in range(len(columns)):
            col_vals = [r[i] for r in rows if len(r) > i and isinstance(r[i], (int, float))]
            if col_vals:
                data_cols.append(col_vals)
        if data_cols:
            ax.boxplot(data_cols, tick_labels=columns[:len(data_cols)])

    def _plot_heatmap(self, ax, columns, rows, spec):
        import numpy as np
        data = []
        for r in rows:
            row_nums = [v if isinstance(v, (int, float)) else 0 for v in r]
            data.append(row_nums)
        arr = np.array(data, dtype=float) if data else np.zeros((1, 1))
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.figure.colorbar(im, ax=ax)


# ══════════════════════════════════════════════════════════
#  Mermaid Diagram Renderer
# ══════════════════════════════════════════════════════════

class MermaidDiagramRenderer(RendererProtocol):
    """Renders Mermaid DSL diagrams.

    Strategy:
    1. If mmdc CLI is available → subprocess render (best quality)
    2. Fallback → save raw .mmd file + text-based preview
    """

    def __init__(self, mmdc_path: Optional[str] = None):
        self._mmdc_path = mmdc_path or "mmdc"

    async def render(
        self,
        spec: Dict[str, Any],
        data: Optional[Dict[str, Any]],
        constraints: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Render a mermaid diagram."""
        dsl = spec.get("dsl", "")
        if not dsl:
            return {"status": "error", "error": "No DSL provided", "artifacts": []}

        os.makedirs(output_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        output_format = constraints.get("output_format", "svg")

        # Try mmdc subprocess
        try:
            result = await self._render_mmdc(dsl, output_dir, ts, output_format)
            if result["status"] == "ok":
                return result
        except Exception as e:
            logger.debug(f"mmdc not available: {e}")

        # Fallback: save raw .mmd + text preview
        return self._render_fallback(dsl, output_dir, ts)

    async def _render_mmdc(self, dsl: str, output_dir: str, ts: int, fmt: str) -> Dict[str, Any]:
        """Attempt mmdc subprocess rendering."""
        import asyncio

        mmd_path = os.path.join(output_dir, f"diagram_{ts}.mmd")
        out_path = os.path.join(output_dir, f"diagram_{ts}.{fmt}")

        with open(mmd_path, "w") as f:
            f.write(dsl)

        proc = await asyncio.create_subprocess_exec(
            self._mmdc_path, "-i", mmd_path, "-o", out_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            return {"status": "error", "error": stderr.decode()[:500], "artifacts": []}

        size_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        content_hash = ""
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        return {
            "status": "ok",
            "engine": "mermaid",
            "artifacts": [{
                "uri": out_path,
                "format": fmt,
                "size_bytes": size_bytes,
                "width_px": 0,
                "height_px": 0,
                "content_hash": content_hash,
            }],
        }

    def _render_fallback(self, dsl: str, output_dir: str, ts: int) -> Dict[str, Any]:
        """Fallback: save .mmd source file."""
        mmd_path = os.path.join(output_dir, f"diagram_{ts}.mmd")
        with open(mmd_path, "w") as f:
            f.write(dsl)

        size_bytes = os.path.getsize(mmd_path)
        with open(mmd_path, "rb") as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        return {
            "status": "ok",
            "engine": "mermaid_fallback",
            "artifacts": [{
                "uri": mmd_path,
                "format": "mmd",
                "size_bytes": size_bytes,
                "width_px": 0,
                "height_px": 0,
                "content_hash": content_hash,
            }],
            "note": "mmdc not available — raw .mmd file saved. Install @mermaid-js/mermaid-cli for SVG/PNG.",
        }


# ══════════════════════════════════════════════════════════
#  Fake Renderer (for offline testing)
# ══════════════════════════════════════════════════════════

class FakeRenderer(RendererProtocol):
    """Deterministic fake renderer for testing. Returns predictable artifacts."""

    def __init__(self, engine_name: str = "fake"):
        self.engine_name = engine_name
        self.render_calls: List[Dict[str, Any]] = []

    async def render(
        self,
        spec: Dict[str, Any],
        data: Optional[Dict[str, Any]],
        constraints: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Return a deterministic fake result."""
        self.render_calls.append({
            "spec": spec, "data": data,
            "constraints": constraints, "output_dir": output_dir,
        })

        output_format = constraints.get("output_format", "png")
        os.makedirs(output_dir, exist_ok=True)

        # Create a minimal file
        filename = f"fake_{self.engine_name}_{len(self.render_calls)}.{output_format}"
        filepath = os.path.join(output_dir, filename)

        content = f"FAKE:{self.engine_name}:{output_format}".encode()
        with open(filepath, "wb") as f:
            f.write(content)

        content_hash = hashlib.sha256(content).hexdigest()[:16]

        return {
            "status": "ok",
            "engine": self.engine_name,
            "artifacts": [{
                "uri": filepath,
                "format": output_format,
                "size_bytes": len(content),
                "width_px": 800,
                "height_px": 600,
                "content_hash": content_hash,
            }],
        }
