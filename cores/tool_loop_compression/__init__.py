"""Tool Loop Compression Module (Wave C). Rule #1: __init__ NON importa adapter
top-level (hot-path core import pulls backend on mcp-server → crash). Adapter LAZY."""

from pathlib import Path
__version__ = "1.0.0"
__all__ = ["create_module"]


def create_module(module_path: Path, **kwargs):
    from .adapter import ToolLoopCompressionAdapter
    return ToolLoopCompressionAdapter(module_path, **kwargs)
