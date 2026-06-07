"""
Sub-Layer Zero — Layer 0 Snapshot Management.

Handles generation, validation, and sliding window management
for Layer 0 (Working Memory) snapshots.
"""

import logging
from typing import Any, Dict, List, Optional

from .models import SubLayerZeroSnapshot

logger = logging.getLogger(__name__)


class SubLayerZeroManager:
    """
    Manages Layer 0 (Working Memory) — a sliding window of snapshots.

    Each snapshot captures the contextual state at a specific turn.
    The window size is configurable (default: 5 snapshots).
    """

    def __init__(self, max_snapshots: int = 5):
        """
        Initialize Sub-Layer Zero manager.

        Args:
            max_snapshots: Maximum number of snapshots in the sliding window.
        """
        self.max_snapshots = max_snapshots

    def add_snapshot(
        self,
        current_snapshots: List[Dict[str, Any]],
        new_snapshot: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Add a new snapshot to the sliding window.

        If the window is full, the oldest snapshot is evicted.

        Args:
            current_snapshots: Existing list of snapshot dicts.
            new_snapshot: New snapshot to add.

        Returns:
            Updated list of snapshots (max_snapshots length).
        """
        validated = self.validate_snapshot(new_snapshot)
        snapshots = list(current_snapshots)
        snapshots.append(validated)

        # Sliding window — evict oldest if over limit
        if len(snapshots) > self.max_snapshots:
            overflow = len(snapshots) - self.max_snapshots
            evicted = snapshots[:overflow]
            snapshots = snapshots[overflow:]
            logger.debug(
                f"[SubLayerZero] Evicted {len(evicted)} snapshot(s), "
                f"window size: {len(snapshots)}/{self.max_snapshots}"
            )

        return snapshots

    def validate_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize a snapshot using the Pydantic model.

        Missing optional fields are filled with defaults.

        Args:
            snapshot: Raw snapshot dict.

        Returns:
            Validated snapshot dict.

        Raises:
            ValueError: If required fields are missing.
        """
        try:
            model = SubLayerZeroSnapshot(**snapshot)
            return model.model_dump()
        except Exception as e:
            logger.warning(f"[SubLayerZero] Snapshot validation error: {e}")
            raise ValueError(f"Invalid snapshot: {e}") from e

    def should_compress(
        self, snapshots: List[Dict[str, Any]], compress_at: int = 4
    ) -> bool:
        """
        Check whether the compression trigger threshold is reached.

        Args:
            snapshots: Current Layer 0 snapshots.
            compress_at: Number of snapshots that triggers compression (N-1).

        Returns:
            True if compression should be triggered.
        """
        return len(snapshots) >= compress_at

    def get_snapshots_for_compression(
        self, snapshots: List[Dict[str, Any]], batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Get the snapshots to feed into the compression engine.

        Returns the oldest `batch_size` snapshots from the window.

        Args:
            snapshots: Current Layer 0 snapshots.
            batch_size: Number of snapshots per compression batch.

        Returns:
            List of snapshots for compression.
        """
        return snapshots[:batch_size]
