"""
Collection Manager Providers - Pure Technical Logic

Pure provider implementations with zero framework dependencies.
Can be tested independently and reused in other contexts.
"""

from typing import Dict, Any, Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class DatabaseProvider(Protocol):
    """
    Protocol defining interface contract for database providers.

    Benefits:
    - Type safety at development time
    - Clear contract for implementations
    - Runtime validation with isinstance()
    - IDE autocomplete support
    """

    async def health_check(self) -> str:
        """
        Check database health.

        Returns:
            Status string: "healthy" or "unhealthy"
        """
        ...

    async def close(self) -> None:
        """Close database connection."""
        ...


class MockDBClient:
    """
    Mock database client for development and testing.

    In production, this would be replaced with actual PostgreSQL client.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mock database client.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.connected = True
        logger.info("MockDBClient initialized")

    async def health_check(self) -> str:
        """
        Check database health.

        Returns:
            "healthy" if connected, "unhealthy" otherwise
        """
        return "healthy" if self.connected else "unhealthy"

    async def close(self) -> None:
        """Close database connection."""
        self.connected = False
        logger.info("MockDBClient connection closed")
