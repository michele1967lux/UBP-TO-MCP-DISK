"""
streaming_rag/handlers.py

Protocol-specific handlers for streaming.

Provides:
- SSEHandler: Server-Sent Events for HTTP
- WebSocketHandler: WebSocket connections
- HTTPChunkedHandler: HTTP chunked transfer
- ResponseBuilder: Build streaming responses

v1.0.0: Initial release
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

from .providers import (
    StreamEvent,
    TokenEvent,
    ChunkEvent,
    EventType,
    StreamState,
    StreamStatus,
    OutputFormat,
    SSEConfig,
    WebSocketConfig,
    TimeoutConfig,
)
from .streams import (
    BaseStream,
    SSEStream,
    WebSocketStream,
    JSONLStream,
    TextStream,
    StreamFactory,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Response Types
# ============================================================================


@dataclass
class StreamingResponse:
    """Streaming response wrapper."""
    content_type: str
    headers: Dict[str, str] = field(default_factory=dict)
    status_code: int = 200
    stream: Optional[BaseStream] = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get all response headers."""
        base_headers = {
            "Content-Type": self.content_type,
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
        base_headers.update(self.headers)
        return base_headers


# ============================================================================
# Base Handler
# ============================================================================


class BaseHandler(ABC):
    """Base class for protocol handlers."""
    
    def __init__(
        self,
        stream_factory: StreamFactory,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        self._stream_factory = stream_factory
        self._timeout_config = timeout_config or TimeoutConfig()
        self._active_streams: Dict[str, BaseStream] = {}
    
    @abstractmethod
    async def handle(
        self,
        stream_id: str,
        state: StreamState,
        **kwargs,
    ) -> StreamingResponse:
        """Handle streaming request."""
        pass
    
    def get_stream(self, stream_id: str) -> Optional[BaseStream]:
        """Get active stream by ID."""
        return self._active_streams.get(stream_id)
    
    async def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream."""
        stream = self._active_streams.get(stream_id)
        if stream:
            stream.state.is_cancelled = True
            stream.state.status = StreamStatus.CANCELLED
            await stream.close()
            del self._active_streams[stream_id]
            return True
        return False
    
    def cleanup_stream(self, stream_id: str) -> None:
        """Clean up stream after completion."""
        if stream_id in self._active_streams:
            del self._active_streams[stream_id]


# ============================================================================
# SSE Handler
# ============================================================================


class SSEHandler(BaseHandler):
    """Handler for Server-Sent Events."""
    
    def __init__(
        self,
        stream_factory: StreamFactory,
        sse_config: Optional[SSEConfig] = None,
        **kwargs,
    ):
        super().__init__(stream_factory, **kwargs)
        self._sse_config = sse_config or SSEConfig()
    
    async def handle(
        self,
        stream_id: str,
        state: StreamState,
        **kwargs,
    ) -> StreamingResponse:
        """Create SSE streaming response."""
        state.format = OutputFormat.SSE
        
        stream = self._stream_factory.create(
            format=OutputFormat.SSE,
            stream_id=stream_id,
            state=state,
        )
        
        self._active_streams[stream_id] = stream
        
        return StreamingResponse(
            content_type="text/event-stream",
            headers={
                "X-Stream-ID": stream_id,
            },
            stream=stream,
        )
    
    async def iterate_sse(
        self,
        stream: BaseStream,
    ) -> AsyncGenerator[bytes, None]:
        """Iterate SSE stream as bytes."""
        async for chunk in stream.iterate():
            yield chunk.encode("utf-8")


# ============================================================================
# WebSocket Handler
# ============================================================================


class WebSocketHandler(BaseHandler):
    """Handler for WebSocket connections."""
    
    def __init__(
        self,
        stream_factory: StreamFactory,
        ws_config: Optional[WebSocketConfig] = None,
        **kwargs,
    ):
        super().__init__(stream_factory, **kwargs)
        self._ws_config = ws_config or WebSocketConfig()
    
    async def handle(
        self,
        stream_id: str,
        state: StreamState,
        websocket: Any = None,
        **kwargs,
    ) -> StreamingResponse:
        """Handle WebSocket connection."""
        state.format = OutputFormat.WEBSOCKET
        
        stream = self._stream_factory.create(
            format=OutputFormat.WEBSOCKET,
            stream_id=stream_id,
            state=state,
        )
        
        if isinstance(stream, WebSocketStream) and websocket:
            stream.set_websocket(websocket)
        
        self._active_streams[stream_id] = stream
        
        return StreamingResponse(
            content_type="application/json",
            headers={
                "X-Stream-ID": stream_id,
            },
            stream=stream,
        )
    
    async def handle_messages(
        self,
        stream_id: str,
        websocket: Any,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Handle incoming WebSocket messages."""
        stream = self._active_streams.get(stream_id)
        if not stream:
            return
        
        try:
            async for message in websocket:
                if message.type == 1:  # TEXT
                    data = json.loads(message.data)
                    
                    if data.get("type") == "cancel":
                        await self.cancel_stream(stream_id)
                        break
                    elif data.get("type") == "ping":
                        await websocket.send_str(json.dumps({"type": "pong"}))
                    elif on_message:
                        on_message(message.data)
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.cleanup_stream(stream_id)


# ============================================================================
# HTTP Chunked Handler
# ============================================================================


class HTTPChunkedHandler(BaseHandler):
    """Handler for HTTP chunked transfer encoding."""
    
    async def handle(
        self,
        stream_id: str,
        state: StreamState,
        format: OutputFormat = OutputFormat.JSONL,
        **kwargs,
    ) -> StreamingResponse:
        """Create chunked HTTP response."""
        state.format = format
        
        stream = self._stream_factory.create(
            format=format,
            stream_id=stream_id,
            state=state,
        )
        
        self._active_streams[stream_id] = stream
        
        content_type = {
            OutputFormat.JSONL: "application/x-ndjson",
            OutputFormat.NDJSON: "application/x-ndjson",
            OutputFormat.TEXT: "text/plain",
        }.get(format, "application/octet-stream")
        
        return StreamingResponse(
            content_type=content_type,
            headers={
                "Transfer-Encoding": "chunked",
                "X-Stream-ID": stream_id,
            },
            stream=stream,
        )


# ============================================================================
# Response Builder
# ============================================================================


class ResponseBuilder:
    """
    Builds streaming responses for different frameworks.
    
    Supports:
    - FastAPI/Starlette
    - aiohttp
    - Generic ASGI
    """
    
    def __init__(
        self,
        sse_handler: SSEHandler,
        ws_handler: WebSocketHandler,
        http_handler: HTTPChunkedHandler,
    ):
        self._sse = sse_handler
        self._ws = ws_handler
        self._http = http_handler
    
    async def build_sse_response(
        self,
        stream_id: str,
        state: StreamState,
    ) -> Dict[str, Any]:
        """Build SSE response for FastAPI."""
        response = await self._sse.handle(stream_id, state)
        
        return {
            "media_type": response.content_type,
            "headers": response.get_headers(),
            "stream": response.stream,
        }
    
    async def build_jsonl_response(
        self,
        stream_id: str,
        state: StreamState,
    ) -> Dict[str, Any]:
        """Build JSON Lines response."""
        response = await self._http.handle(
            stream_id,
            state,
            format=OutputFormat.JSONL,
        )
        
        return {
            "media_type": response.content_type,
            "headers": response.get_headers(),
            "stream": response.stream,
        }
    
    async def build_text_response(
        self,
        stream_id: str,
        state: StreamState,
    ) -> Dict[str, Any]:
        """Build plain text streaming response."""
        response = await self._http.handle(
            stream_id,
            state,
            format=OutputFormat.TEXT,
        )
        
        return {
            "media_type": response.content_type,
            "headers": response.get_headers(),
            "stream": response.stream,
        }
    
    def create_fastapi_response(
        self,
        stream: BaseStream,
        media_type: str,
        headers: Dict[str, str],
    ):
        """
        Create FastAPI StreamingResponse.
        
        Note: Import FastAPI dependencies at runtime to avoid hard dependency.
        """
        try:
            from starlette.responses import StreamingResponse
            
            async def generate():
                async for chunk in stream.iterate():
                    if isinstance(chunk, str):
                        yield chunk.encode("utf-8")
                    else:
                        yield chunk
            
            return StreamingResponse(
                generate(),
                media_type=media_type,
                headers=headers,
            )
        except ImportError:
            raise RuntimeError("FastAPI/Starlette not installed")
    
    def create_aiohttp_response(
        self,
        stream: BaseStream,
        content_type: str,
        headers: Dict[str, str],
    ):
        """
        Create aiohttp StreamResponse.
        
        Note: Import aiohttp at runtime.
        """
        try:
            from aiohttp import web
            
            response = web.StreamResponse(
                status=200,
                headers=headers,
            )
            response.content_type = content_type
            return response, stream
        except ImportError:
            raise RuntimeError("aiohttp not installed")


# ============================================================================
# Connection Manager
# ============================================================================


class ConnectionManager:
    """Manages active streaming connections."""
    
    def __init__(self, max_connections: int = 100):
        self._max_connections = max_connections
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def add_connection(
        self,
        stream_id: str,
        stream: BaseStream,
        client_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a new connection."""
        async with self._lock:
            if len(self._connections) >= self._max_connections:
                return False
            
            self._connections[stream_id] = {
                "stream": stream,
                "client_info": client_info or {},
                "connected_at": datetime.utcnow(),
            }
            return True
    
    async def remove_connection(self, stream_id: str) -> None:
        """Remove a connection."""
        async with self._lock:
            if stream_id in self._connections:
                del self._connections[stream_id]
    
    async def broadcast(self, event: StreamEvent, exclude: Optional[List[str]] = None) -> int:
        """Broadcast event to all connections."""
        exclude = exclude or []
        count = 0
        
        async with self._lock:
            for stream_id, conn in self._connections.items():
                if stream_id not in exclude:
                    try:
                        await conn["stream"].emit(event)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Broadcast failed for {stream_id}: {e}")
        
        return count
    
    def get_connection_count(self) -> int:
        """Get current connection count."""
        return len(self._connections)
    
    def get_connection_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get connection info."""
        conn = self._connections.get(stream_id)
        if conn:
            return {
                "stream_id": stream_id,
                "client_info": conn["client_info"],
                "connected_at": conn["connected_at"].isoformat(),
                "stream_status": conn["stream"].state.status.value,
            }
        return None
    
    async def cleanup_stale(self, max_idle_seconds: int = 60) -> int:
        """Clean up stale connections."""
        now = datetime.utcnow()
        to_remove = []
        
        async with self._lock:
            for stream_id, conn in self._connections.items():
                stream = conn["stream"]
                if stream.is_closed:
                    to_remove.append(stream_id)
                elif stream.state.status == StreamStatus.COMPLETED:
                    to_remove.append(stream_id)
                elif stream.state.status == StreamStatus.ERROR:
                    to_remove.append(stream_id)
        
        for stream_id in to_remove:
            await self.remove_connection(stream_id)
        
        return len(to_remove)


# ============================================================================
# LLM Streaming Adapter
# ============================================================================


class LLMStreamingAdapter:
    """
    Adapts LLM module streaming to our stream format.
    
    Handles:
    - Token-by-token passthrough
    - Chunk aggregation
    - Error recovery
    """
    
    def __init__(
        self,
        llm_module: Any,
        operation: str = "stream_generate",
    ):
        self._llm = llm_module
        self._operation = operation
    
    async def stream_tokens(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from LLM."""
        try:
            # Check if module supports streaming
            if hasattr(self._llm, self._operation):
                stream_method = getattr(self._llm, self._operation)
                async for token in stream_method(prompt=prompt, **kwargs):
                    if isinstance(token, dict):
                        yield token.get("token", token.get("text", ""))
                    else:
                        yield str(token)
            elif hasattr(self._llm, "generate"):
                # Fallback to non-streaming with simulated streaming
                result = await self._llm.generate(prompt=prompt, **kwargs)
                text = result.get("text", "") if isinstance(result, dict) else str(result)
                
                # Simulate streaming by yielding words
                for word in text.split():
                    yield word + " "
                    await asyncio.sleep(0.01)  # Small delay
            else:
                raise RuntimeError("LLM module does not support generation")
                
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise
    
    async def stream_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream generation with RAG context."""
        # Build prompt
        context_text = "\n\n".join([
            f"[Source {i+1}]: {doc.get('content', doc.get('text', ''))[:500]}"
            for i, doc in enumerate(context[:5])
        ])
        
        prompt = f"""Context:
{context_text}

Question: {query}

Answer based on the context provided:"""
        
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        async for token in self.stream_tokens(prompt=prompt, **kwargs):
            yield token
