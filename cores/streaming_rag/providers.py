"""
streaming_rag/providers.py

Core data classes and streaming primitives.

Provides:
- StreamEvent: Base event class
- TokenEvent: Token generation event
- RetrievalEvent: Retrieval progress event
- ErrorEvent: Error event
- StreamBuffer: Token buffering
- StreamState: Stream state management
- StreamMetrics: Performance metrics
- BackpressureController: Flow control

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, AsyncGenerator, AsyncIterator, Callable, Deque,
    Dict, Generic, List, Optional, Protocol, TypeVar, Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class EventType(Enum):
    """Stream event types."""
    # Lifecycle
    STREAM_START = "stream_start"
    STREAM_END = "stream_end"
    STREAM_ERROR = "stream_error"
    
    # Retrieval
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_PROGRESS = "retrieval_progress"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    SOURCE_FOUND = "source_found"
    
    # Generation
    GENERATION_START = "generation_start"
    TOKEN = "token"
    CHUNK = "chunk"
    GENERATION_PROGRESS = "generation_progress"
    GENERATION_COMPLETE = "generation_complete"
    
    # Metadata
    METADATA = "metadata"
    TIMING = "timing"
    
    # Control
    HEARTBEAT = "heartbeat"
    CANCEL = "cancel"
    DONE = "done"


class StreamStatus(Enum):
    """Stream status."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class OutputFormat(Enum):
    """Output format types."""
    SSE = "sse"
    WEBSOCKET = "websocket"
    JSONL = "jsonl"
    NDJSON = "ndjson"
    TEXT = "text"
    GENERATOR = "generator"


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class TokenStreamConfig:
    """Token streaming configuration."""
    chunk_size: int = 1
    min_chunk_tokens: int = 1
    max_chunk_tokens: int = 50
    flush_interval_ms: int = 50
    word_boundary_flush: bool = True


@dataclass
class BufferConfig:
    """Buffer configuration."""
    enabled: bool = True
    buffer_tokens: int = 5
    buffer_timeout_ms: int = 100
    max_buffer_size: int = 1000
    overflow_strategy: str = "drop_oldest"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = False
    tokens_per_second: int = 100
    burst_size: int = 50
    delay_between_chunks_ms: int = 0


@dataclass
class SSEConfig:
    """Server-Sent Events configuration."""
    enabled: bool = True
    event_prefix: str = "rag"
    keep_alive_interval_ms: int = 15000
    retry_timeout_ms: int = 3000
    include_id: bool = True


@dataclass
class WebSocketConfig:
    """WebSocket configuration."""
    enabled: bool = True
    ping_interval_ms: int = 30000
    pong_timeout_ms: int = 10000
    max_message_size: int = 65536


@dataclass
class BackpressureConfig:
    """Backpressure configuration."""
    enabled: bool = True
    high_watermark: int = 80
    low_watermark: int = 20
    pause_on_pressure: bool = True


@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    stream_timeout: int = 300
    idle_timeout: int = 60
    first_token_timeout: int = 30
    retrieval_timeout: int = 30


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    track_token_latency: bool = True
    track_ttft: bool = True
    track_throughput: bool = True


# ============================================================================
# Event Classes
# ============================================================================


@dataclass
class StreamEvent:
    """Base stream event."""
    event_id: str
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def to_sse(self, prefix: str = "rag") -> str:
        """Convert to SSE format."""
        import json
        event_name = f"{prefix}_{self.event_type.value}"
        data = json.dumps(self.to_dict())
        return f"event: {event_name}\nid: {self.event_id}\ndata: {data}\n\n"
    
    def to_jsonl(self) -> str:
        """Convert to JSON Lines format."""
        import json
        return json.dumps(self.to_dict()) + "\n"


@dataclass
class TokenEvent(StreamEvent):
    """Token generation event."""
    token: str = ""
    token_index: int = 0
    is_final: bool = False
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        self.event_type = EventType.TOKEN
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "token": self.token,
            "token_index": self.token_index,
            "is_final": self.is_final,
        })
        return base


@dataclass
class ChunkEvent(StreamEvent):
    """Chunk generation event (multiple tokens)."""
    content: str = ""
    start_index: int = 0
    end_index: int = 0
    token_count: int = 0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        self.event_type = EventType.CHUNK
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "token_count": self.token_count,
        })
        return base


@dataclass
class RetrievalEvent(StreamEvent):
    """Retrieval progress event."""
    stage: str = ""  # start, searching, found, complete
    documents_found: int = 0
    total_expected: int = 0
    query: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "stage": self.stage,
            "documents_found": self.documents_found,
            "total_expected": self.total_expected,
            "query": self.query,
        })
        return base


@dataclass
class SourceEvent(StreamEvent):
    """Source document found event."""
    source_id: str = ""
    title: str = ""
    content_preview: str = ""
    relevance_score: float = 0.0
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        self.event_type = EventType.SOURCE_FOUND
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "source_id": self.source_id,
            "title": self.title,
            "content_preview": self.content_preview[:200],
            "relevance_score": round(self.relevance_score, 3),
            "source_metadata": self.source_metadata,
        })
        return base


@dataclass
class ProgressEvent(StreamEvent):
    """Progress update event."""
    phase: str = ""  # retrieval, generation
    progress_percent: float = 0.0
    tokens_generated: int = 0
    estimated_remaining_ms: int = 0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "phase": self.phase,
            "progress_percent": round(self.progress_percent, 1),
            "tokens_generated": self.tokens_generated,
            "estimated_remaining_ms": self.estimated_remaining_ms,
        })
        return base


@dataclass
class TimingEvent(StreamEvent):
    """Timing information event."""
    phase: str = ""
    duration_ms: float = 0.0
    ttft_ms: Optional[float] = None  # Time to first token
    tokens_per_second: Optional[float] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        self.event_type = EventType.TIMING
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "phase": self.phase,
            "duration_ms": round(self.duration_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms else None,
            "tokens_per_second": round(self.tokens_per_second, 1) if self.tokens_per_second else None,
        })
        return base


@dataclass
class ErrorEvent(StreamEvent):
    """Error event."""
    error_code: str = ""
    error_message: str = ""
    recoverable: bool = False
    partial_result: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        self.event_type = EventType.STREAM_ERROR
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "error_code": self.error_code,
            "error_message": self.error_message,
            "recoverable": self.recoverable,
            "has_partial_result": self.partial_result is not None,
        })
        return base


@dataclass
class DoneEvent(StreamEvent):
    """Stream completion event."""
    total_tokens: int = 0
    total_sources: int = 0
    total_duration_ms: float = 0.0
    final_answer: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        self.event_type = EventType.DONE
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "total_tokens": self.total_tokens,
            "total_sources": self.total_sources,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "final_answer_length": len(self.final_answer),
        })
        return base


# ============================================================================
# Stream Buffer
# ============================================================================


class StreamBuffer:
    """
    Token buffer with configurable flushing.
    
    Supports:
    - Token accumulation
    - Timed flush
    - Word boundary flush
    - Overflow handling
    """
    
    def __init__(self, config: BufferConfig):
        self.config = config
        self._buffer: Deque[str] = deque()
        self._total_tokens = 0
        self._last_flush = time.perf_counter()
        self._lock = asyncio.Lock()
    
    async def add(self, token: str) -> Optional[str]:
        """Add token to buffer, return flushed content if threshold reached."""
        async with self._lock:
            self._buffer.append(token)
            self._total_tokens += 1
            
            # Check overflow
            if len(self._buffer) >= self.config.max_buffer_size:
                if self.config.overflow_strategy == "drop_oldest":
                    self._buffer.popleft()
                elif self.config.overflow_strategy == "flush":
                    return self._do_flush()
            
            # Check flush conditions
            if self._should_flush(token):
                return self._do_flush()
            
            return None
    
    def _should_flush(self, latest_token: str) -> bool:
        """Check if buffer should be flushed."""
        # Token count threshold
        if len(self._buffer) >= self.config.buffer_tokens:
            return True
        
        # Time threshold
        elapsed_ms = (time.perf_counter() - self._last_flush) * 1000
        if elapsed_ms >= self.config.buffer_timeout_ms:
            return True
        
        return False
    
    def _do_flush(self) -> str:
        """Flush buffer and return content."""
        content = "".join(self._buffer)
        self._buffer.clear()
        self._last_flush = time.perf_counter()
        return content
    
    async def flush(self) -> str:
        """Force flush buffer."""
        async with self._lock:
            return self._do_flush()
    
    @property
    def size(self) -> int:
        return len(self._buffer)
    
    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# ============================================================================
# Stream State
# ============================================================================


@dataclass
class StreamState:
    """State for an active stream."""
    stream_id: str
    session_id: str
    query: str
    status: StreamStatus = StreamStatus.PENDING
    format: OutputFormat = OutputFormat.SSE
    
    # Progress
    tokens_generated: int = 0
    sources_found: int = 0
    current_phase: str = ""
    
    # Timing
    started_at: Optional[datetime] = None
    first_token_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Content
    accumulated_content: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    
    # Control
    is_cancelled: bool = False
    error: Optional[str] = None
    
    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.started_at and self.first_token_at:
            return (self.first_token_at - self.started_at).total_seconds() * 1000
        return None
    
    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        if not self.started_at:
            return 0.0
        
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds() * 1000
    
    @property
    def tokens_per_second(self) -> float:
        """Token generation rate."""
        if self.tokens_generated == 0:
            return 0.0
        
        duration_s = self.duration_ms / 1000
        if duration_s == 0:
            return 0.0
        
        return self.tokens_generated / duration_s
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "format": self.format.value,
            "tokens_generated": self.tokens_generated,
            "sources_found": self.sources_found,
            "current_phase": self.current_phase,
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms else None,
            "duration_ms": round(self.duration_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "is_cancelled": self.is_cancelled,
            "error": self.error,
        }


# ============================================================================
# Backpressure Controller
# ============================================================================


class BackpressureController:
    """Controls flow based on consumer capacity."""
    
    def __init__(self, config: BackpressureConfig):
        self.config = config
        self._queue_size = 0
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
    
    def update_queue_size(self, size: int) -> None:
        """Update current queue size."""
        self._queue_size = size
        
        if not self.config.enabled:
            return
        
        if size >= self.config.high_watermark and not self._paused:
            self._paused = True
            self._pause_event.clear()
            logger.debug(f"Backpressure: paused at {size}")
        elif size <= self.config.low_watermark and self._paused:
            self._paused = False
            self._pause_event.set()
            logger.debug(f"Backpressure: resumed at {size}")
    
    async def wait_if_paused(self) -> None:
        """Wait if backpressure is active."""
        if self.config.enabled and self.config.pause_on_pressure:
            await self._pause_event.wait()
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    @property
    def pressure_level(self) -> float:
        """Current pressure level (0-100)."""
        return min(100, (self._queue_size / self.config.high_watermark) * 100)


# ============================================================================
# Stream Metrics
# ============================================================================


class StreamMetrics:
    """Collects streaming metrics."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._total_streams = 0
        self._completed_streams = 0
        self._failed_streams = 0
        self._total_tokens = 0
        self._ttft_values: List[float] = []
        self._throughput_values: List[float] = []
        self._token_latencies: List[float] = []
    
    def record_stream_start(self) -> None:
        """Record stream started."""
        self._total_streams += 1
    
    def record_stream_complete(
        self,
        tokens: int,
        ttft_ms: Optional[float],
        throughput: float,
    ) -> None:
        """Record stream completed."""
        self._completed_streams += 1
        self._total_tokens += tokens
        
        if ttft_ms and self.config.track_ttft:
            self._ttft_values.append(ttft_ms)
            if len(self._ttft_values) > 1000:
                self._ttft_values = self._ttft_values[-1000:]
        
        if self.config.track_throughput:
            self._throughput_values.append(throughput)
            if len(self._throughput_values) > 1000:
                self._throughput_values = self._throughput_values[-1000:]
    
    def record_stream_error(self) -> None:
        """Record stream error."""
        self._failed_streams += 1
    
    def record_token_latency(self, latency_ms: float) -> None:
        """Record inter-token latency."""
        if self.config.track_token_latency:
            self._token_latencies.append(latency_ms)
            if len(self._token_latencies) > 10000:
                self._token_latencies = self._token_latencies[-10000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "total_streams": self._total_streams,
            "completed_streams": self._completed_streams,
            "failed_streams": self._failed_streams,
            "success_rate": (
                self._completed_streams / max(self._total_streams, 1)
            ),
            "total_tokens_generated": self._total_tokens,
            "ttft_stats": {
                "avg_ms": sum(self._ttft_values) / len(self._ttft_values) if self._ttft_values else 0,
                "min_ms": min(self._ttft_values) if self._ttft_values else 0,
                "max_ms": max(self._ttft_values) if self._ttft_values else 0,
            },
            "throughput_stats": {
                "avg_tokens_per_sec": sum(self._throughput_values) / len(self._throughput_values) if self._throughput_values else 0,
            },
            "token_latency_stats": {
                "avg_ms": sum(self._token_latencies) / len(self._token_latencies) if self._token_latencies else 0,
            },
        }
