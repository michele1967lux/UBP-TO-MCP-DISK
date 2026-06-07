"""
Batch Ingestion Job - Data Model

Server-side batch ingestion state management.
Follows the same pattern as report_session.py (Redis-persisted state with enums + dataclasses).

Redis Keys:
- ubp:ingest:job:{job_id}                  - Job state + per-file manifest (JSON, TTL 3600s)
- ubp:ingest:data:{job_id}:{idx}           - File content at index (STRING, TTL 3600s)
- ubp:ingest:active:{user_id}:{collection} - Active job pointer for reconnect (STRING, TTL 3600s)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class IngestJobState(str, Enum):
    """State of the batch ingestion job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileStatus(str, Enum):
    """Status of a single file within the batch."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"


@dataclass
class IngestFileEntry:
    """Metadata for a single file in the batch."""

    filename: str
    file_type: str
    size: int
    status: FileStatus = FileStatus.PENDING
    document_id: Optional[str] = None
    chunks_count: int = 0
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "size": self.size,
            "status": self.status.value,
            "document_id": self.document_id,
            "chunks_count": self.chunks_count,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestFileEntry":
        return cls(
            filename=data["filename"],
            file_type=data["file_type"],
            size=data["size"],
            status=FileStatus(data.get("status", "pending")),
            document_id=data.get("document_id"),
            chunks_count=data.get("chunks_count", 0),
            error=data.get("error"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class IngestJob:
    """Complete batch ingestion job state."""

    job_id: str
    user_id: str
    collection_id: str
    state: IngestJobState
    files: List[IngestFileEntry]
    created_at: str
    updated_at: str
    current_file_index: int = 0
    success_count: int = 0
    error_count: int = 0
    skip_count: int = 0
    chunking_config: Optional[Dict[str, Any]] = None

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def progress_pct(self) -> float:
        if self.total_files == 0:
            return 0.0
        done = self.success_count + self.error_count + self.skip_count
        # Count skipped files too (cancelled files)
        skipped = sum(1 for f in self.files if f.status == FileStatus.SKIPPED)
        return ((done + skipped) / self.total_files) * 100.0

    @property
    def current_filename(self) -> Optional[str]:
        if 0 <= self.current_file_index < self.total_files:
            return self.files[self.current_file_index].filename
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "collection_id": self.collection_id,
            "state": self.state.value,
            "files": [f.to_dict() for f in self.files],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_file_index": self.current_file_index,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "skip_count": self.skip_count,
            "chunking_config": self.chunking_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestJob":
        return cls(
            job_id=data["job_id"],
            user_id=data["user_id"],
            collection_id=data["collection_id"],
            state=IngestJobState(data["state"]),
            files=[IngestFileEntry.from_dict(f) for f in data.get("files", [])],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            current_file_index=data.get("current_file_index", 0),
            success_count=data.get("success_count", 0),
            error_count=data.get("error_count", 0),
            skip_count=data.get("skip_count", 0),
            chunking_config=data.get("chunking_config"),
        )

    def to_status_response(self) -> Dict[str, Any]:
        """Build the polling response dict."""
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "collection_id": self.collection_id,
            "total_files": self.total_files,
            "current_file_index": self.current_file_index,
            "current_filename": self.current_filename,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "skip_count": self.skip_count,
            "progress_pct": round(self.progress_pct, 1),
            "files": [f.to_dict() for f in self.files],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
