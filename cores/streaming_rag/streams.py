"""
streaming_rag/streams.py

Stream implementations for different protocols.

Provides:
- BaseStream: Abstract stream interface
- AsyncGeneratorStream: Python async generator
- SSEStream: Server-Sent Events
- WebSocketStream: WebSocket protocol
- JSONLStream: JSON Lines format
- TextStream: Plain text streaming

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from .providers import (
    StreamEvent,
    TokenEvent,
    ChunkEvent,
    RetrievalEvent,
    SourceEvent,
    ProgressEvent,
    TimingEvent,
    ErrorEvent,
    DoneEvent,
    EventType,
    StreamState,
    StreamStatus,
    OutputFormat,
    StreamBuffer,
    BufferConfig,
    BackpressureController,
    BackpressureConfig,
    SSEConfig,
    WebSocketConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Base Stream
# ============================================================================


class BaseStream(ABC):
    """Abstract base class for streams."""
    
    def __init__(
        self,
        stream_id: str,
        state: StreamState,
        buffer_config: Optional[BufferConfig] = None,
        backpressure_config: Optional[BackpressureConfig] = None,
    ):
        self.stream_id = stream_id
        self.state = state
        self._buffer = StreamBuffer(buffer_config or BufferConfig())
        self._backpressure = BackpressureController(backpressure_config or BackpressureConfig())
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._event_id_counter = 0
    
    def _next_event_id(self) -> str:
        """Generate next event ID."""
        self._event_id_counter += 1
        return f"{self.stream_id}_{self._event_id_counter}"
    
    async def emit(self, event: StreamEvent) -> None:
        """Emit an event to the stream."""
        if self._closed:
            return
        
        # Wait for backpressure
        await self._backpressure.wait_if_paused()
        
        await self._event_queue.put(event)
        self._backpressure.update_queue_size(self._event_queue.qsize())
    
    async def emit_token(self, token: str, index: int, is_final: bool = False) -> None:
        """Emit a token event."""
        # Update state
        if self.state.first_token_at is None:
            self.state.first_token_at = __import__("datetime").datetime.utcnow()
        
        self.state.tokens_generated += 1
        self.state.accumulated_content += token
        
        event = TokenEvent(
            event_id=self._next_event_id(),
            event_type=EventType.TOKEN,
            token=token,
            token_index=index,
            is_final=is_final,
        )
        await self.emit(event)
    
    async def emit_chunk(self, content: str, start: int, end: int) -> None:
        """Emit a chunk event (multiple tokens)."""
        token_count = end - start
        self.state.tokens_generated += token_count
        self.state.accumulated_content += content
        
        if self.state.first_token_at is None:
            self.state.first_token_at = __import__("datetime").datetime.utcnow()
        
        event = ChunkEvent(
            event_id=self._next_event_id(),
            event_type=EventType.CHUNK,
            content=content,
            start_index=start,
            end_index=end,
            token_count=token_count,
        )
        await self.emit(event)
    
    async def emit_retrieval(
        self,
        event_type: EventType,
        stage: str,
        docs_found: int = 0,
        total: int = 0,
        query: str = "",
    ) -> None:
        """Emit a retrieval event."""
        self.state.current_phase = "retrieval"
        
        event = RetrievalEvent(
            event_id=self._next_event_id(),
            event_type=event_type,
            stage=stage,
            documents_found=docs_found,
            total_expected=total,
            query=query,
        )
        await self.emit(event)
    
    async def emit_source(
        self,
        source_id: str,
        title: str,
        content: str,
        score: float,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Emit a source found event."""
        self.state.sources_found += 1
        self.state.sources.append({
            "id": source_id,
            "title": title,
            "score": score,
        })
        
        event = SourceEvent(
            event_id=self._next_event_id(),
            event_type=EventType.SOURCE_FOUND,
            source_id=source_id,
            title=title,
            content_preview=content[:200] if content else "",
            relevance_score=score,
            source_metadata=metadata or {},
        )
        await self.emit(event)
    
    async def emit_progress(
        self,
        phase: str,
        percent: float,
        tokens: int = 0,
        estimated_remaining_ms: int = 0,
    ) -> None:
        """Emit progress event."""
        self.state.current_phase = phase
        
        event = ProgressEvent(
            event_id=self._next_event_id(),
            event_type=EventType.GENERATION_PROGRESS,
            phase=phase,
            progress_percent=percent,
            tokens_generated=tokens,
            estimated_remaining_ms=estimated_remaining_ms,
        )
        await self.emit(event)
    
    async def emit_timing(
        self,
        phase: str,
        duration_ms: float,
        ttft_ms: Optional[float] = None,
        tps: Optional[float] = None,
    ) -> None:
        """Emit timing event."""
        event = TimingEvent(
            event_id=self._next_event_id(),
            event_type=EventType.TIMING,
            phase=phase,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
            tokens_per_second=tps,
        )
        await self.emit(event)
    
    async def emit_error(
        self,
        code: str,
        message: str,
        recoverable: bool = False,
        partial_result: Optional[str] = None,
    ) -> None:
        """Emit error event."""
        self.state.status = StreamStatus.ERROR
        self.state.error = message
        
        event = ErrorEvent(
            event_id=self._next_event_id(),
            event_type=EventType.STREAM_ERROR,
            error_code=code,
            error_message=message,
            recoverable=recoverable,
            partial_result=partial_result,
        )
        await self.emit(event)
    
    async def emit_done(self) -> None:
        """Emit done event and close stream."""
        self.state.status = StreamStatus.COMPLETED
        self.state.completed_at = __import__("datetime").datetime.utcnow()
        
        event = DoneEvent(
            event_id=self._next_event_id(),
            event_type=EventType.DONE,
            total_tokens=self.state.tokens_generated,
            total_sources=self.state.sources_found,
            total_duration_ms=self.state.duration_ms,
            final_answer=self.state.accumulated_content,
        )
        await self.emit(event)
        await self.close()
    
    async def close(self) -> None:
        """Close the stream."""
        self._closed = True
        await self._event_queue.put(None)  # Signal end
    
    @abstractmethod
    async def iterate(self) -> AsyncGenerator[Any, None]:
        """Iterate over stream events."""
        pass
    
    @property
    def is_closed(self) -> bool:
        return self._closed


# ============================================================================
# Async Generator Stream
# ============================================================================


class AsyncGeneratorStream(BaseStream):
    """Stream that yields events as Python objects."""
    
    async def iterate(self) -> AsyncGenerator[StreamEvent, None]:
        """Iterate over events as StreamEvent objects."""
        while not self._closed or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                
                if event is None:
                    break
                
                yield event
                
            except asyncio.TimeoutError:
                if self._closed:
                    break
                continue
    
    async def iterate_tokens(self) -> AsyncGenerator[str, None]:
        """Iterate over just the token text."""
        async for event in self.iterate():
            if isinstance(event, TokenEvent):
                yield event.token
            elif isinstance(event, ChunkEvent):
                yield event.content


# ============================================================================
# SSE Stream
# ============================================================================


class SSEStream(BaseStream):
    """Server-Sent Events stream."""
    
    def __init__(
        self,
        stream_id: str,
        state: StreamState,
        sse_config: Optional[SSEConfig] = None,
        **kwargs,
    ):
        super().__init__(stream_id, state, **kwargs)
        self._sse_config = sse_config or SSEConfig()
        self._last_keepalive = time.perf_counter()
    
    def _format_sse(self, event: StreamEvent) -> str:
        """Format event as SSE."""
        lines = []
        
        # Event type
        event_name = f"{self._sse_config.event_prefix}_{event.event_type.value}"
        lines.append(f"event: {event_name}")
        
        # Event ID
        if self._sse_config.include_id:
            lines.append(f"id: {event.event_id}")
        
        # Data
        data = json.dumps(event.to_dict())
        lines.append(f"data: {data}")
        
        # Empty line to end event
        lines.append("")
        lines.append("")
        
        return "\n".join(lines)
    
    def _keepalive(self) -> str:
        """Generate keepalive comment."""
        return ": keepalive\n\n"
    
    async def iterate(self) -> AsyncGenerator[str, None]:
        """Iterate over events as SSE formatted strings."""
        # Send retry timeout
        yield f"retry: {self._sse_config.retry_timeout_ms}\n\n"
        
        keepalive_interval_s = self._sse_config.keep_alive_interval_ms / 1000
        
        while not self._closed or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=min(1.0, keepalive_interval_s),
                )
                
                if event is None:
                    break
                
                yield self._format_sse(event)
                self._last_keepalive = time.perf_counter()
                
            except asyncio.TimeoutError:
                # Check if keepalive needed
                if time.perf_counter() - self._last_keepalive >= keepalive_interval_s:
                    yield self._keepalive()
                    self._last_keepalive = time.perf_counter()
                
                if self._closed:
                    break


# ============================================================================
# WebSocket Stream
# ============================================================================


class WebSocketStream(BaseStream):
    """WebSocket stream."""
    
    def __init__(
        self,
        stream_id: str,
        state: StreamState,
        ws_config: Optional[WebSocketConfig] = None,
        **kwargs,
    ):
        super().__init__(stream_id, state, **kwargs)
        self._ws_config = ws_config or WebSocketConfig()
        self._websocket: Optional[Any] = None
    
    def set_websocket(self, ws: Any) -> None:
        """Set the WebSocket connection."""
        self._websocket = ws
    
    def _format_message(self, event: StreamEvent) -> str:
        """Format event as WebSocket message."""
        return json.dumps({
            "type": "event",
            "event": event.to_dict(),
        })
    
    async def iterate(self) -> AsyncGenerator[str, None]:
        """Iterate over events as WebSocket messages."""
        while not self._closed or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                
                if event is None:
                    break
                
                yield self._format_message(event)
                
            except asyncio.TimeoutError:
                if self._closed:
                    break
    
    async def send_to_ws(self) -> None:
        """Send events to WebSocket (if connected)."""
        if not self._websocket:
            return
        
        async for message in self.iterate():
            try:
                await self._websocket.send_str(message)
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                break


# ============================================================================
# JSON Lines Stream
# ============================================================================


class JSONLStream(BaseStream):
    """JSON Lines (newline-delimited JSON) stream."""
    
    async def iterate(self) -> AsyncGenerator[str, None]:
        """Iterate over events as JSON lines."""
        while not self._closed or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                
                if event is None:
                    break
                
                yield json.dumps(event.to_dict()) + "\n"
                
            except asyncio.TimeoutError:
                if self._closed:
                    break


# ============================================================================
# Text Stream
# ============================================================================


class TextStream(BaseStream):
    """Plain text stream (tokens only)."""
    
    async def iterate(self) -> AsyncGenerator[str, None]:
        """Iterate over just the text content."""
        while not self._closed or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                
                if event is None:
                    break
                
                # Only yield text content
                if isinstance(event, TokenEvent):
                    yield event.token
                elif isinstance(event, ChunkEvent):
                    yield event.content
                
            except asyncio.TimeoutError:
                if self._closed:
                    break


# ============================================================================
# Stream Factory
# ============================================================================


class StreamFactory:
    """Factory for creating streams."""
    
    def __init__(
        self,
        buffer_config: Optional[BufferConfig] = None,
        backpressure_config: Optional[BackpressureConfig] = None,
        sse_config: Optional[SSEConfig] = None,
        ws_config: Optional[WebSocketConfig] = None,
    ):
        self._buffer_config = buffer_config or BufferConfig()
        self._backpressure_config = backpressure_config or BackpressureConfig()
        self._sse_config = sse_config or SSEConfig()
        self._ws_config = ws_config or WebSocketConfig()
    
    def create(
        self,
        format: OutputFormat,
        stream_id: str,
        state: StreamState,
    ) -> BaseStream:
        """Create a stream of the specified format."""
        kwargs = {
            "stream_id": stream_id,
            "state": state,
            "buffer_config": self._buffer_config,
            "backpressure_config": self._backpressure_config,
        }
        
        if format == OutputFormat.SSE:
            return SSEStream(sse_config=self._sse_config, **kwargs)
        elif format == OutputFormat.WEBSOCKET:
            return WebSocketStream(ws_config=self._ws_config, **kwargs)
        elif format == OutputFormat.JSONL or format == OutputFormat.NDJSON:
            return JSONLStream(**kwargs)
        elif format == OutputFormat.TEXT:
            return TextStream(**kwargs)
        else:  # GENERATOR or default
            return AsyncGeneratorStream(**kwargs)


# ============================================================================
# Streaming Pipeline
# ============================================================================


class StreamingPipeline:
    """
    Orchestrates streaming RAG pipeline.
    
    Combines retrieval and generation with streaming output.
    """
    
    def __init__(
        self,
        stream: BaseStream,
        retrieval_callback: Optional[Callable] = None,
        generation_callback: Optional[Callable] = None,
    ):
        self.stream = stream
        self._retrieval_callback = retrieval_callback
        self._generation_callback = generation_callback
    
    async def stream_retrieval(
        self,
        retrieval_coro,
        emit_sources: bool = True,
    ) -> List[Dict[str, Any]]:
        """Stream retrieval phase."""
        await self.stream.emit_retrieval(
            EventType.RETRIEVAL_START,
            stage="start",
            query=self.stream.state.query,
        )
        
        try:
            results = await retrieval_coro
            
            if emit_sources and results:
                for i, doc in enumerate(results):
                    await self.stream.emit_source(
                        source_id=doc.get("id", f"doc_{i}"),
                        title=doc.get("title", doc.get("metadata", {}).get("title", "")),
                        content=doc.get("content", doc.get("text", "")),
                        score=doc.get("score", 0.0),
                        metadata=doc.get("metadata", {}),
                    )
            
            await self.stream.emit_retrieval(
                EventType.RETRIEVAL_COMPLETE,
                stage="complete",
                docs_found=len(results) if results else 0,
                query=self.stream.state.query,
            )
            
            return results or []
            
        except Exception as e:
            await self.stream.emit_error(
                code="RETRIEVAL_ERROR",
                message=str(e),
                recoverable=True,
            )
            return []
    
    async def stream_generation(
        self,
        generation_async_iter: AsyncGenerator[str, None],
        emit_progress: bool = True,
        progress_interval: int = 50,
    ) -> str:
        """Stream generation phase."""
        await self.stream.emit(StreamEvent(
            event_id=self.stream._next_event_id(),
            event_type=EventType.GENERATION_START,
        ))
        
        self.stream.state.current_phase = "generation"
        accumulated = ""
        token_index = 0
        
        try:
            async for token in generation_async_iter:
                await self.stream.emit_token(token, token_index)
                accumulated += token
                token_index += 1
                
                # Progress events
                if emit_progress and token_index % progress_interval == 0:
                    await self.stream.emit_progress(
                        phase="generation",
                        percent=min(100, token_index / 10),  # Rough estimate
                        tokens=token_index,
                    )
            
            await self.stream.emit(StreamEvent(
                event_id=self.stream._next_event_id(),
                event_type=EventType.GENERATION_COMPLETE,
            ))
            
            return accumulated
            
        except Exception as e:
            await self.stream.emit_error(
                code="GENERATION_ERROR",
                message=str(e),
                recoverable=False,
                partial_result=accumulated if accumulated else None,
            )
            return accumulated
    
    async def run(
        self,
        retrieval_coro,
        generation_factory: Callable[[List[Dict[str, Any]]], AsyncGenerator[str, None]],
    ) -> None:
        """Run complete streaming pipeline."""
        self.stream.state.status = StreamStatus.ACTIVE
        self.stream.state.started_at = __import__("datetime").datetime.utcnow()
        
        try:
            # Retrieval phase
            sources = await self.stream_retrieval(retrieval_coro)
            
            # Generation phase
            gen_iter = generation_factory(sources)
            await self.stream_generation(gen_iter)
            
            # Emit timing
            await self.stream.emit_timing(
                phase="total",
                duration_ms=self.stream.state.duration_ms,
                ttft_ms=self.stream.state.ttft_ms,
                tps=self.stream.state.tokens_per_second,
            )
            
            # Done
            await self.stream.emit_done()
            
        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}")
            await self.stream.emit_error(
                code="PIPELINE_ERROR",
                message=str(e),
                partial_result=self.stream.state.accumulated_content or None,
            )
            await self.stream.close()
