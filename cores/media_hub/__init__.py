"""media_hub module — Factory entry point (3-file pattern)."""

from pathlib import Path
from typing import Any, Optional


def create_module(
    module_path: Path = None,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
    container: Optional[Any] = None,
    config: Optional[dict] = None,
):
    """Create MediaHubAdapter instance (DI factory).

    Args:
        module_path: Path to module directory (from module_loader)
        di_container: DI container (from module_loader)
        event_bus: Event bus (from module_loader)
        container: Legacy alias for di_container
        config: Optional config dict
    """
    from .adapter import MediaHubAdapter
    return MediaHubAdapter(
        container=di_container or container,
        config=config,
    )


# Backward-compat alias
create_adapter = create_module
