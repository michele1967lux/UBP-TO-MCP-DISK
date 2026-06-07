"""
Artifact Manager for Report Export (v2.5 - FEAT-ARTIFACT-001)

This module handles the generation, storage, and retrieval of report artifacts
(DOCX, Markdown files) from completed report sessions.

Architecture:
- Artifacts are stored on disk at /app/artifacts (Docker volume)
- Metadata is stored in Redis for fast lookup and access control
- Download URLs are generated for secure file retrieval

Redis Keys:
- ubp:{env}:artifact:{artifact_id} - Artifact metadata (JSON)
- ubp:{env}:artifact:session:{session_id} - List of artifacts for a session

File Structure:
- /app/artifacts/{user_id}/{artifact_id}.{ext}
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ArtifactFormat(str, Enum):
    """Supported artifact formats."""
    DOCX = "docx"
    MARKDOWN = "md"
    PDF = "pdf"  # Future support


@dataclass
class ArtifactMetadata:
    """Metadata for a generated artifact."""
    artifact_id: str
    session_id: str
    user_id: str
    filename: str
    format: str
    size_bytes: int
    title: str
    created_at: str
    expires_at: str
    download_url: str
    storage_path: str
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactMetadata":
        """Create from dictionary."""
        return cls(**data)


class ArtifactManager:
    """
    Manager for artifact generation and storage.

    Responsibilities:
    - Save generated artifacts to disk
    - Store metadata in Redis for lookup
    - Generate secure download URLs
    - Handle artifact expiration
    - Validate user access to artifacts

    Usage:
        manager = ArtifactManager(redis_client, base_path="/app/artifacts")
        metadata = await manager.save_artifact(
            content=docx_bytes,
            format="docx",
            session_id="...",
            user_id="...",
            title="Report Title"
        )
        # Returns ArtifactMetadata with download_url
    """

    # Configuration
    DEFAULT_BASE_PATH = "/app/artifacts"
    DEFAULT_TTL_HOURS = 24 * 7  # 7 days default expiration
    REDIS_KEY_PREFIX = "ubp:artifact"

    def __init__(
        self,
        redis_client,
        base_path: Optional[str] = None,
        base_url: str = "/api/artifacts",
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ):
        """
        Initialize ArtifactManager.

        Args:
            redis_client: Async Redis client for metadata storage
            base_path: Base directory for artifact storage
            base_url: Base URL for download links
            ttl_hours: Artifact expiration time in hours
        """
        self._redis = redis_client
        self._base_path = Path(base_path or self.DEFAULT_BASE_PATH)
        self._base_url = base_url
        self._ttl_hours = ttl_hours
        self._cleanup_task = None  # v6.4.1: Periodic cleanup task

        # Ensure base directory exists
        self._ensure_directory(self._base_path)

        logger.info(
            f"[ARTIFACT] ArtifactManager initialized: base_path={self._base_path}, "
            f"ttl={self._ttl_hours}h"
        )

    def _ensure_directory(self, path: Path) -> None:
        """Ensure directory exists, create if needed."""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"[ARTIFACT] Could not create directory {path}: {e}")

    def _get_redis_key(self, artifact_id: str) -> str:
        """Get Redis key for artifact metadata."""
        return f"{self.REDIS_KEY_PREFIX}:{artifact_id}"

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session artifacts list."""
        return f"{self.REDIS_KEY_PREFIX}:session:{session_id}"

    def _get_user_index_key(self, user_id: str) -> str:
        """Get Redis key for user artifacts index (SET)."""
        return f"ubp:user:{user_id}:artifacts"

    def _generate_filename(
        self,
        session_id: str,
        format: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Generate unique filename for artifact.

        Format: report_{session_id_short}_{timestamp}.{ext}
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_short = session_id[:8]

        # Sanitize title if provided
        if title:
            # Remove special chars, limit length
            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
            safe_title = safe_title.strip().replace(" ", "_")[:30]
            if safe_title:
                return f"report_{safe_title}_{timestamp}.{format}"

        return f"report_{session_short}_{timestamp}.{format}"

    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate SHA-256 checksum of content."""
        import hashlib
        return hashlib.sha256(content).hexdigest()

    async def save_artifact(
        self,
        content: bytes,
        format: str,
        session_id: str,
        user_id: str,
        title: str = "Report",
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> ArtifactMetadata:
        """
        Save artifact to disk and register in Redis.

        Args:
            content: Binary content of the artifact (DOCX bytes, etc.)
            format: File format (docx, md, pdf)
            session_id: Report session ID
            user_id: Owner user ID for access control
            title: Report title for filename generation
            metadata_extra: Additional metadata to store

        Returns:
            ArtifactMetadata with download_url
        """
        # Generate IDs and paths
        artifact_id = str(uuid.uuid4())
        filename = self._generate_filename(session_id, format, title)

        # Create user directory
        user_dir = self._base_path / user_id
        # v6.4.0: Path traversal protection
        try:
            user_dir.resolve().relative_to(self._base_path.resolve())
        except ValueError:
            logger.error(f"[ARTIFACT] Path traversal attempt: user_id='{user_id}'")
            raise ValueError("Invalid user_id: path traversal detected")
        self._ensure_directory(user_dir)

        # Full storage path
        storage_path = user_dir / filename

        # Calculate metadata
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self._ttl_hours)
        checksum = self._calculate_checksum(content)

        # Build download URL
        download_url = f"{self._base_url}/{artifact_id}"

        # Create metadata object
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            session_id=session_id,
            user_id=user_id,
            filename=filename,
            format=format,
            size_bytes=len(content),
            title=title,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            download_url=download_url,
            storage_path=str(storage_path),
            checksum=checksum,
        )

        try:
            # Write file to disk
            with open(storage_path, "wb") as f:
                f.write(content)

            logger.info(
                f"[ARTIFACT] Saved artifact to disk: {storage_path} "
                f"({len(content)} bytes, checksum={checksum[:8]}...)"
            )

            # Store metadata in Redis
            redis_key = self._get_redis_key(artifact_id)
            ttl_seconds = self._ttl_hours * 3600

            await self._redis.setex(
                redis_key,
                ttl_seconds,
                json.dumps(metadata.to_dict()),
            )

            # Add to session's artifact list
            session_key = self._get_session_key(session_id)
            await self._redis.rpush(session_key, artifact_id)
            await self._redis.expire(session_key, ttl_seconds)

            # Add to user's artifact index (v2.6 FEAT-ARTIFACT-LIST)
            user_index_key = self._get_user_index_key(user_id)
            await self._redis.sadd(user_index_key, artifact_id)
            await self._redis.expire(user_index_key, ttl_seconds)

            logger.info(
                f"[ARTIFACT] Registered artifact in Redis: {artifact_id} "
                f"(session={session_id[:8]}..., user={user_id[:8]}..., expires={expires_at.isoformat()})"
            )

            return metadata

        except Exception as e:
            logger.error(f"[ARTIFACT] Failed to save artifact: {e}")
            # Cleanup partial writes
            if storage_path.exists():
                try:
                    storage_path.unlink()
                except Exception:
                    pass
            raise

    async def get_artifact(
        self,
        artifact_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[ArtifactMetadata]:
        """
        Get artifact metadata by ID.

        Args:
            artifact_id: Artifact UUID
            user_id: If provided, validates ownership

        Returns:
            ArtifactMetadata or None if not found/unauthorized
        """
        redis_key = self._get_redis_key(artifact_id)

        try:
            data = await self._redis.get(redis_key)
            if not data:
                logger.warning(f"[ARTIFACT] Artifact not found: {artifact_id}")
                return None

            metadata = ArtifactMetadata.from_dict(json.loads(data))

            # Validate ownership if user_id provided
            if user_id and metadata.user_id != user_id:
                logger.warning(
                    f"[ARTIFACT] Access denied: artifact {artifact_id} "
                    f"owned by {metadata.user_id}, requested by {user_id}"
                )
                return None

            return metadata

        except Exception as e:
            logger.error(f"[ARTIFACT] Error getting artifact {artifact_id}: {e}")
            return None

    async def get_artifact_file(
        self,
        artifact_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[tuple[bytes, ArtifactMetadata]]:
        """
        Get artifact file content and metadata.

        Args:
            artifact_id: Artifact UUID
            user_id: If provided, validates ownership

        Returns:
            Tuple of (file_bytes, metadata) or None
        """
        metadata = await self.get_artifact(artifact_id, user_id)
        if not metadata:
            return None

        storage_path = Path(metadata.storage_path)

        if not storage_path.exists():
            logger.error(f"[ARTIFACT] File not found on disk: {storage_path}")
            return None

        try:
            with open(storage_path, "rb") as f:
                content = f.read()

            # Verify checksum
            if metadata.checksum:
                actual_checksum = self._calculate_checksum(content)
                if actual_checksum != metadata.checksum:
                    logger.error(
                        f"[ARTIFACT] Checksum mismatch for {artifact_id}: "
                        f"expected {metadata.checksum}, got {actual_checksum}"
                    )
                    return None

            return content, metadata

        except Exception as e:
            logger.error(f"[ARTIFACT] Error reading file {storage_path}: {e}")
            return None

    async def list_session_artifacts(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> List[ArtifactMetadata]:
        """
        List all artifacts for a report session.

        Args:
            session_id: Report session ID
            user_id: If provided, filters by ownership

        Returns:
            List of ArtifactMetadata objects
        """
        session_key = self._get_session_key(session_id)

        try:
            artifact_ids = await self._redis.lrange(session_key, 0, -1)
            if not artifact_ids:
                return []

            artifacts = []
            for aid in artifact_ids:
                aid_str = aid.decode() if isinstance(aid, bytes) else aid
                metadata = await self.get_artifact(aid_str, user_id)
                if metadata:
                    artifacts.append(metadata)

            return artifacts

        except Exception as e:
            logger.error(f"[ARTIFACT] Error listing session artifacts: {e}")
            return []

    async def list_user_artifacts(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[ArtifactMetadata]:
        """
        List all artifacts for a user (v2.6 FEAT-ARTIFACT-LIST).

        Uses the user index SET (ubp:user:{user_id}:artifacts) for fast lookup.
        Falls back to empty list for users without indexed artifacts.

        Args:
            user_id: User ID to list artifacts for
            limit: Maximum number of artifacts to return

        Returns:
            List of ArtifactMetadata objects sorted by created_at (newest first)
        """
        user_index_key = self._get_user_index_key(user_id)

        try:
            # Get artifact IDs from user index SET
            artifact_ids = await self._redis.smembers(user_index_key)
            if not artifact_ids:
                logger.debug(f"[ARTIFACT] No artifacts found for user {user_id[:8]}...")
                return []

            artifacts = []
            for aid in artifact_ids:
                aid_str = aid.decode() if isinstance(aid, bytes) else aid
                metadata = await self.get_artifact(aid_str, user_id)
                if metadata:
                    artifacts.append(metadata)

            # Sort by created_at descending (newest first)
            artifacts.sort(key=lambda a: a.created_at, reverse=True)

            # Apply limit
            if limit and len(artifacts) > limit:
                artifacts = artifacts[:limit]

            logger.info(
                f"[ARTIFACT] Listed {len(artifacts)} artifacts for user {user_id[:8]}..."
            )
            return artifacts

        except Exception as e:
            logger.error(f"[ARTIFACT] Error listing user artifacts: {e}")
            return []

    async def delete_artifact(
        self,
        artifact_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Delete an artifact (file and metadata).

        Args:
            artifact_id: Artifact UUID
            user_id: If provided, validates ownership

        Returns:
            True if deleted successfully
        """
        metadata = await self.get_artifact(artifact_id, user_id)
        if not metadata:
            return False

        try:
            # Delete file
            storage_path = Path(metadata.storage_path)
            if storage_path.exists():
                storage_path.unlink()
                logger.info(f"[ARTIFACT] Deleted file: {storage_path}")

            # Delete Redis metadata
            redis_key = self._get_redis_key(artifact_id)
            await self._redis.delete(redis_key)

            # Remove from session list
            session_key = self._get_session_key(metadata.session_id)
            await self._redis.lrem(session_key, 0, artifact_id)

            # Remove from user index (v2.6 FEAT-ARTIFACT-LIST)
            user_index_key = self._get_user_index_key(metadata.user_id)
            await self._redis.srem(user_index_key, artifact_id)

            logger.info(f"[ARTIFACT] Deleted artifact: {artifact_id}")
            return True

        except Exception as e:
            logger.error(f"[ARTIFACT] Error deleting artifact {artifact_id}: {e}")
            return False

    async def cleanup_expired(self) -> int:
        """
        Clean up expired artifacts from disk.

        Note: Redis automatically expires keys, but files on disk
        need manual cleanup. Run this periodically.

        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        now = datetime.now(timezone.utc)

        try:
            # Walk through artifact directories
            for user_dir in self._base_path.iterdir():
                if not user_dir.is_dir():
                    continue

                for artifact_file in user_dir.iterdir():
                    if not artifact_file.is_file():
                        continue

                    # Check file age (use mtime as proxy for expiration)
                    mtime = datetime.fromtimestamp(artifact_file.stat().st_mtime)
                    age_hours = (now - mtime).total_seconds() / 3600

                    if age_hours > self._ttl_hours:
                        try:
                            artifact_file.unlink()
                            cleaned += 1
                            logger.info(f"[ARTIFACT] Cleaned up expired: {artifact_file}")
                        except Exception as e:
                            logger.warning(f"[ARTIFACT] Could not delete {artifact_file}: {e}")

            if cleaned > 0:
                logger.info(f"[ARTIFACT] Cleanup completed: {cleaned} files removed")

            return cleaned

        except Exception as e:
            logger.error(f"[ARTIFACT] Cleanup error: {e}")
            return cleaned

    async def start_periodic_cleanup(self, interval_seconds: int = 3600):
        """Start a background task that periodically cleans expired artifacts."""
        import asyncio
        if self._cleanup_task and not self._cleanup_task.done():
            logger.debug("[ARTIFACT] Periodic cleanup already running")
            return
        self._cleanup_task = asyncio.create_task(
            self._periodic_cleanup_loop(interval_seconds),
            name="artifact_cleanup",
        )
        logger.info(f"[ARTIFACT] Scheduled periodic cleanup every {interval_seconds}s")

    async def _periodic_cleanup_loop(self, interval: int):
        """Background loop for periodic cleanup."""
        import asyncio
        while True:
            try:
                await asyncio.sleep(interval)
                cleaned = await self.cleanup_expired()
                if cleaned:
                    logger.info(f"[ARTIFACT] Periodic cleanup removed {cleaned} expired artifacts")
            except asyncio.CancelledError:
                logger.info("[ARTIFACT] Periodic cleanup task cancelled")
                break
            except Exception as e:
                logger.warning(f"[ARTIFACT] Periodic cleanup error: {e}")
