"""Tool Loop Compression Module Factory — 3-file pattern. Adapter import LAZY (Rule #1)."""
from pathlib import Path
from typing import Any, Optional


def create_module(module_path: Path, di_container: Optional[Any] = None, event_bus: Optional[Any] = None, **kwargs):
    from .adapter import ToolLoopCompressionAdapter
    return ToolLoopCompressionAdapter(module_path=module_path, di_container=di_container, event_bus=event_bus)
