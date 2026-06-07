"""
streaming_rag/adapter.py

Bridge layer that exposes all streaming RAG operations to the UBP system.

Operations:
- initialize: Start components
- stream_query: Full streaming RAG pipeline
- stream_sse: SSE streaming endpoint
- stream_websocket: WebSocket streaming
- stream_jsonl: JSON Lines streaming
- stream_text: Plain text streaming
- create_stream: Create a stream manually
- emit_token: Emit token to stream
- emit_event: Emit custom event
- cancel_stream: Cancel active stream
- get_stream_state: Get stream state
- list_streams: List active streams
- get_stats: Get metrics (admin)
- reload_config: Hot-reload (admin)
- shutdown: Graceful shutdown
- health_check: Component health

v1.0.0: Initial release with full enterprise features

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import (
    Any, AsyncGenerator, Callable, Dict, List, 
    Optional, Protocol, Tuple, Union,
)

import sys
from pathlib import Path as _Path
_portable_path = str(_Path(__file__).resolve().parent.parent.parent)
if _portable_path not in sys.path:
    sys.path.insert(0, _portable_path)

from _portable.context import PortableContext

from .providers import (
    # Events
    StreamEvent,
    TokenEvent,
    ChunkEvent,
    RetrievalEvent,
    SourceEvent,
    ProgressEvent,
    TimingEvent,
    ErrorEvent,
    DoneEvent,
    # Enums
    EventType,
    StreamStatus,
    OutputFormat,
    # Configs
    TokenStreamConfig,
    BufferConfig,
    RateLimitConfig,
    SSEConfig,
    WebSocketConfig,
    BackpressureConfig,
    TimeoutConfig,
    MetricsConfig,
    # Providers
    StreamState,
    StreamBuffer,
    BackpressureController,
    StreamMetrics,
)
from .streams import (
    BaseStream,
    SSEStream,
    WebSocketStream,
    JSONLStream,
    TextStream,
    AsyncGeneratorStream,
    StreamFactory,
    StreamingPipeline,
)
from .handlers import (
    SSEHandler,
    WebSocketHandler,
    HTTPChunkedHandler,
    ResponseBuilder,
    ConnectionManager,
    LLMStreamingAdapter,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# DI Container Wrapper
# ============================================================================


class DIContainerModuleRegistry:
    """Wraps DI container to provide module registry interface."""

    def __init__(self, di_container: Optional[Any] = None):
        self._container = di_container
        self._cached_modules: Dict[str, Any] = {}

    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a module by name (sync - cache only)."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]
        return None

    async def resolve_module(self, module_name: str) -> Optional[Any]:
        """Async module resolution via DI container."""
        if module_name in self._cached_modules:
            return self._cached_modules[module_name]

        if not self._container:
            return None

        # DI container.resolve() is async - must be awaited
        if hasattr(self._container, "resolve"):
            try:
                module = await self._container.resolve(module_name)
                if module:
                    self._cached_modules[module_name] = module
                    return module
            except Exception as e:
                logger.warning(f"Failed to resolve module '{module_name}': {e}")

        return None


# ============================================================================
# Configuration Utilities
# ============================================================================


def resolve_env_value(value: Any) -> Any:
    """Resolve environment variable placeholders."""
    if not isinstance(value, str):
        return value
    
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace(match):
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    
    return re.sub(pattern, replace, value)


def coerce_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively coerce configuration values."""
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = coerce_config_types(value)
        elif isinstance(value, list):
            result[key] = [
                coerce_config_types(v) if isinstance(v, dict) else _coerce_value(v)
                for v in value
            ]
        else:
            result[key] = _coerce_value(value)
    
    return result


def _coerce_value(value: Any) -> Any:
    """Coerce a single value."""
    if not isinstance(value, str):
        return value
    
    value = resolve_env_value(value)
    
    if not isinstance(value, str):
        return value
    
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False
    
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


# ============================================================================
# Streaming RAG Adapter
# ============================================================================


class StreamingRAGAdapter:
    """
    Adapter exposing streaming RAG operations.
    
    Features:
    - Multiple output formats (SSE, WebSocket, JSONL, Text)
    - Token-by-token streaming
    - Chunk buffering
    - Backpressure handling
    - Progress events
    - Integration with retrieval and LLM modules
    """
    
    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self._di_container = di_container
        self._event_bus = event_bus
        
        self._module_registry = DIContainerModuleRegistry(di_container)
        
        # Configuration
        self._config: Dict[str, Any] = {}
        self._token_config: Optional[TokenStreamConfig] = None
        self._buffer_config: Optional[BufferConfig] = None
        self._rate_config: Optional[RateLimitConfig] = None
        self._sse_config: Optional[SSEConfig] = None
        self._ws_config: Optional[WebSocketConfig] = None
        self._backpressure_config: Optional[BackpressureConfig] = None
        self._timeout_config: Optional[TimeoutConfig] = None
        self._metrics_config: Optional[MetricsConfig] = None
        
        # Components
        self._stream_factory: Optional[StreamFactory] = None
        self._sse_handler: Optional[SSEHandler] = None
        self._ws_handler: Optional[WebSocketHandler] = None
        self._http_handler: Optional[HTTPChunkedHandler] = None
        self._response_builder: Optional[ResponseBuilder] = None
        self._connection_manager: Optional[ConnectionManager] = None
        self._metrics: Optional[StreamMetrics] = None
        
        # Active streams
        self._active_streams: Dict[str, BaseStream] = {}
        self._stream_states: Dict[str, StreamState] = {}
        
        # State
        self._initialized = False

    # ========================================================================
    def _build_context_from_di(self) -> PortableContext:
        return PortableContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="portable",
        )

    def _normalize_ctx(self, ctx: Any) -> PortableContext:
        return PortableContext.normalize(ctx)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        config_path = self.module_path / "config.json"
        
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}")
            return {}
        
        with open(config_path, "r") as f:
            raw_config = json.load(f)
        
        return coerce_config_types(raw_config)
    
    def _build_configs(self) -> None:
        """Build configuration objects."""
        cfg = self._config
        
        token_cfg = cfg.get("token_streaming", {})
        self._token_config = TokenStreamConfig(
            chunk_size=token_cfg.get("chunk_size", 1),
            min_chunk_tokens=token_cfg.get("min_chunk_tokens", 1),
            max_chunk_tokens=token_cfg.get("max_chunk_tokens", 50),
            flush_interval_ms=token_cfg.get("flush_interval_ms", 50),
            word_boundary_flush=token_cfg.get("word_boundary_flush", True),
        )
        
        buffer_cfg = cfg.get("buffering", {})
        self._buffer_config = BufferConfig(
            enabled=buffer_cfg.get("enabled", True),
            buffer_tokens=buffer_cfg.get("buffer_tokens", 5),
            buffer_timeout_ms=buffer_cfg.get("buffer_timeout_ms", 100),
            max_buffer_size=buffer_cfg.get("max_buffer_size", 1000),
            overflow_strategy=buffer_cfg.get("overflow_strategy", "drop_oldest"),
        )
        
        rate_cfg = cfg.get("rate_limiting", {})
        self._rate_config = RateLimitConfig(
            enabled=rate_cfg.get("enabled", False),
            tokens_per_second=rate_cfg.get("tokens_per_second", 100),
            burst_size=rate_cfg.get("burst_size", 50),
            delay_between_chunks_ms=rate_cfg.get("delay_between_chunks_ms", 0),
        )
        
        sse_cfg = cfg.get("sse", {})
        self._sse_config = SSEConfig(
            enabled=sse_cfg.get("enabled", True),
            event_prefix=sse_cfg.get("event_prefix", "rag"),
            keep_alive_interval_ms=sse_cfg.get("keep_alive_interval_ms", 15000),
            retry_timeout_ms=sse_cfg.get("retry_timeout_ms", 3000),
            include_id=sse_cfg.get("include_id", True),
        )
        
        ws_cfg = cfg.get("websocket", {})
        self._ws_config = WebSocketConfig(
            enabled=ws_cfg.get("enabled", True),
            ping_interval_ms=ws_cfg.get("ping_interval_ms", 30000),
            pong_timeout_ms=ws_cfg.get("pong_timeout_ms", 10000),
            max_message_size=ws_cfg.get("max_message_size", 65536),
        )
        
        bp_cfg = cfg.get("backpressure", {})
        self._backpressure_config = BackpressureConfig(
            enabled=bp_cfg.get("enabled", True),
            high_watermark=bp_cfg.get("high_watermark", 80),
            low_watermark=bp_cfg.get("low_watermark", 20),
            pause_on_pressure=bp_cfg.get("pause_on_pressure", True),
        )
        
        timeout_cfg = cfg.get("timeouts", {})
        self._timeout_config = TimeoutConfig(
            stream_timeout=timeout_cfg.get("stream_timeout_seconds", 300),
            idle_timeout=timeout_cfg.get("idle_timeout_seconds", 60),
            first_token_timeout=timeout_cfg.get("first_token_timeout_seconds", 30),
            retrieval_timeout=timeout_cfg.get("retrieval_timeout_seconds", 30),
        )
        
        metrics_cfg = cfg.get("metrics", {})
        self._metrics_config = MetricsConfig(
            enabled=metrics_cfg.get("enabled", True),
            track_token_latency=metrics_cfg.get("track_token_latency", True),
            track_ttft=metrics_cfg.get("track_ttft", True),
            track_throughput=metrics_cfg.get("track_throughput", True),
        )
    
    # ========================================================================
    # Operations
    # ========================================================================
    
    async def initialize(self, ctx: Any = None) -> Dict[str, Any]:
        """Initialize streaming RAG components."""
        if self._initialized:
            return {"status": "already_initialized"}
        
        try:
            self._config = self._load_config()
            self._build_configs()
            
            # Initialize stream factory
            self._stream_factory = StreamFactory(
                buffer_config=self._buffer_config,
                backpressure_config=self._backpressure_config,
                sse_config=self._sse_config,
                ws_config=self._ws_config,
            )
            
            # Initialize handlers
            self._sse_handler = SSEHandler(
                self._stream_factory,
                self._sse_config,
                timeout_config=self._timeout_config,
            )
            
            self._ws_handler = WebSocketHandler(
                self._stream_factory,
                self._ws_config,
                timeout_config=self._timeout_config,
            )
            
            self._http_handler = HTTPChunkedHandler(
                self._stream_factory,
                timeout_config=self._timeout_config,
            )
            
            # Initialize response builder
            self._response_builder = ResponseBuilder(
                self._sse_handler,
                self._ws_handler,
                self._http_handler,
            )
            
            # Initialize connection manager
            max_conn = self._config.get("streaming_rag", {}).get("max_concurrent_streams", 50)
            self._connection_manager = ConnectionManager(max_conn)
            
            # Initialize metrics
            self._metrics = StreamMetrics(self._metrics_config)
            
            self._initialized = True
            
            logger.info("Streaming RAG adapter initialized")
            
            if self._event_bus:
                await self._event_bus.publish(
                    "streaming.initialized",
                    {"module": "streaming_rag", "status": "success"},
                )
            
            return {
                "status": "initialized",
                "formats": ["sse", "websocket", "jsonl", "text"],
            }
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def stream_query(
        self,
        query: str,
        format: str = "sse",
        emit_sources: bool = True,
        emit_progress: bool = True,
        ctx: Any = None,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Stream a RAG query with retrieval and generation.
        
        Args:
            query: User query
            format: Output format (sse, jsonl, text, generator)
            emit_sources: Whether to emit source events
            emit_progress: Whether to emit progress events
            ctx: Security context
        
        Yields:
            Streaming events or text depending on format
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        stream_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Create state
        state = StreamState(
            stream_id=stream_id,
            session_id=session_id,
            query=query,
            format=OutputFormat(format) if format in [f.value for f in OutputFormat] else OutputFormat.SSE,
        )
        
        # Create stream
        output_format = OutputFormat(format) if format in [f.value for f in OutputFormat] else OutputFormat.SSE
        stream = self._stream_factory.create(output_format, stream_id, state)
        
        self._active_streams[stream_id] = stream
        self._stream_states[stream_id] = state
        
        # Record metrics
        if self._metrics:
            self._metrics.record_stream_start()
        
        # Start streaming pipeline in background
        asyncio.create_task(self._run_pipeline(stream, query, emit_sources, emit_progress))
        
        # Yield from stream
        try:
            async for chunk in stream.iterate():
                yield chunk
        finally:
            self._cleanup_stream(stream_id)
    
    async def _run_pipeline(
        self,
        stream: BaseStream,
        query: str,
        emit_sources: bool,
        emit_progress: bool,
    ) -> None:
        """Run the streaming pipeline."""
        try:
            # Retrieval phase
            retrieval_module = self._module_registry.get_module("retrieval_strategy")
            
            if retrieval_module:
                await stream.emit_retrieval(
                    EventType.RETRIEVAL_START,
                    stage="start",
                    query=query,
                )
                
                try:
                    result = await asyncio.wait_for(
                        retrieval_module.retrieve(query=query, top_k=5),
                        timeout=self._timeout_config.retrieval_timeout,
                    )
                    
                    sources = result.get("results", []) if isinstance(result, dict) else []
                    
                    if emit_sources:
                        for i, doc in enumerate(sources[:5]):
                            await stream.emit_source(
                                source_id=doc.get("id", f"doc_{i}"),
                                title=doc.get("title", ""),
                                content=doc.get("content", ""),
                                score=doc.get("score", 0.0),
                            )
                    
                    await stream.emit_retrieval(
                        EventType.RETRIEVAL_COMPLETE,
                        stage="complete",
                        docs_found=len(sources),
                        query=query,
                    )
                    
                except asyncio.TimeoutError:
                    await stream.emit_error(
                        code="RETRIEVAL_TIMEOUT",
                        message="Retrieval timed out",
                        recoverable=True,
                    )
                    sources = []
            else:
                sources = []
            
            # Generation phase
            llm_module = await self._resolve_llm_module()
            
            if llm_module:
                adapter = LLMStreamingAdapter(llm_module)
                
                await stream.emit(StreamEvent(
                    event_id=f"{stream.stream_id}_gen_start",
                    event_type=EventType.GENERATION_START,
                ))
                
                token_index = 0
                
                try:
                    async for token in adapter.stream_with_context(query, sources):
                        await stream.emit_token(token, token_index)
                        token_index += 1
                        
                        if emit_progress and token_index % 50 == 0:
                            await stream.emit_progress(
                                phase="generation",
                                percent=min(100, token_index / 10),
                                tokens=token_index,
                            )
                    
                    await stream.emit(StreamEvent(
                        event_id=f"{stream.stream_id}_gen_complete",
                        event_type=EventType.GENERATION_COMPLETE,
                    ))
                    
                except Exception as e:
                    await stream.emit_error(
                        code="GENERATION_ERROR",
                        message=str(e),
                        partial_result=stream.state.accumulated_content or None,
                    )
            else:
                # Fallback response
                fallback = f"I found {len(sources)} relevant sources for your query: '{query}'"
                for char in fallback:
                    await stream.emit_token(char, 0)
            
            # Timing
            await stream.emit_timing(
                phase="total",
                duration_ms=stream.state.duration_ms,
                ttft_ms=stream.state.ttft_ms,
                tps=stream.state.tokens_per_second,
            )
            
            # Done
            await stream.emit_done()
            
            # Record metrics
            if self._metrics:
                self._metrics.record_stream_complete(
                    tokens=stream.state.tokens_generated,
                    ttft_ms=stream.state.ttft_ms,
                    throughput=stream.state.tokens_per_second,
                )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await stream.emit_error(
                code="PIPELINE_ERROR",
                message=str(e),
            )
            
            if self._metrics:
                self._metrics.record_stream_error()
            
            await stream.close()
    
    async def stream_sse(
        self,
        query: str,
        emit_sources: bool = True,
        ctx: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Stream as Server-Sent Events."""
        async for chunk in self.stream_query(
            query=query,
            format="sse",
            emit_sources=emit_sources,
            ctx=ctx,
        ):
            yield chunk
    
    async def stream_websocket(
        self,
        query: str,
        websocket: Any,
        emit_sources: bool = True,
        ctx: Any = None,
    ) -> None:
        """Stream to WebSocket connection."""
        if not self._initialized:
            await self.initialize(ctx)
        
        stream_id = str(uuid.uuid4())
        
        state = StreamState(
            stream_id=stream_id,
            session_id=str(uuid.uuid4()),
            query=query,
            format=OutputFormat.WEBSOCKET,
        )
        
        response = await self._ws_handler.handle(stream_id, state, websocket=websocket)
        stream = response.stream
        
        self._active_streams[stream_id] = stream
        
        # Run pipeline
        await self._run_pipeline(stream, query, emit_sources, True)
        
        # Send events to WebSocket
        if isinstance(stream, WebSocketStream):
            await stream.send_to_ws()
    
    async def stream_jsonl(
        self,
        query: str,
        emit_sources: bool = True,
        ctx: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Stream as JSON Lines."""
        async for chunk in self.stream_query(
            query=query,
            format="jsonl",
            emit_sources=emit_sources,
            ctx=ctx,
        ):
            yield chunk
    
    async def stream_text(
        self,
        query: str,
        ctx: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Stream as plain text (tokens only)."""
        async for chunk in self.stream_query(
            query=query,
            format="text",
            emit_sources=False,
            emit_progress=False,
            ctx=ctx,
        ):
            yield chunk
    
    async def prepare_stream(
        self,
        query: str,
        mode: str = "token",
        include_sources: bool = True,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Prepare streaming configuration for a query.
        
        Args:
            query: Search query
            mode: Streaming mode (token, chunk, or full)
            include_sources: Whether to include source documents
            ctx: Security context
        
        Returns:
            Stream configuration dict
        """
        if not self._initialized:
            await self.initialize(ctx)
        
        # Generate stream ID and session ID
        stream_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Map mode to format and buffer size
        mode_config = {
            "token": {"format": "sse", "buffer_size": 10},
            "chunk": {"format": "jsonl", "buffer_size": 5},
            "full": {"format": "jsonl", "buffer_size": 1},
        }
        
        # Default to token mode if unknown mode provided
        config_params = mode_config.get(mode, mode_config["token"])
        
        # Prepare stream configuration
        config = {
            "stream_id": stream_id,
            "session_id": session_id,
            "query": query,
            "mode": mode,
            "format": config_params["format"],
            "include_sources": include_sources,
            "buffer_size": config_params["buffer_size"],
            "emit_progress": True,
            "status": "prepared",
        }
        
        return config
    
    async def create_stream(
        self,
        session_id: str,
        query: str,
        format: str = "generator",
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Create a stream manually for custom use."""
        if not self._initialized:
            await self.initialize(ctx)
        
        stream_id = str(uuid.uuid4())
        
        state = StreamState(
            stream_id=stream_id,
            session_id=session_id,
            query=query,
            format=OutputFormat(format) if format in [f.value for f in OutputFormat] else OutputFormat.GENERATOR,
        )
        
        output_format = OutputFormat(format) if format in [f.value for f in OutputFormat] else OutputFormat.GENERATOR
        stream = self._stream_factory.create(output_format, stream_id, state)
        
        self._active_streams[stream_id] = stream
        self._stream_states[stream_id] = state
        
        return {
            "stream_id": stream_id,
            "session_id": session_id,
            "format": format,
            "status": "created",
        }
    
    async def emit_token(
        self,
        stream_id: str,
        token: str,
        index: int = 0,
        is_final: bool = False,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Emit a token to an existing stream."""
        stream = self._active_streams.get(stream_id)
        if not stream:
            return {"error": "Stream not found"}
        
        await stream.emit_token(token, index, is_final)
        
        return {
            "stream_id": stream_id,
            "token_emitted": True,
            "total_tokens": stream.state.tokens_generated,
        }
    
    async def emit_event(
        self,
        stream_id: str,
        event_type: str,
        data: Dict[str, Any],
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Emit a custom event to a stream."""
        stream = self._active_streams.get(stream_id)
        if not stream:
            return {"error": "Stream not found"}
        
        event = StreamEvent(
            event_id=f"{stream_id}_{uuid.uuid4().hex[:8]}",
            event_type=EventType(event_type) if event_type in [e.value for e in EventType] else EventType.METADATA,
            metadata=data,
        )
        
        await stream.emit(event)
        
        return {"stream_id": stream_id, "event_emitted": True}
    
    async def cancel_stream(
        self,
        stream_id: str,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Cancel an active stream."""
        stream = self._active_streams.get(stream_id)
        if not stream:
            return {"error": "Stream not found"}
        
        stream.state.is_cancelled = True
        stream.state.status = StreamStatus.CANCELLED
        await stream.close()
        
        self._cleanup_stream(stream_id)
        
        return {"stream_id": stream_id, "cancelled": True}
    
    async def get_stream_state(
        self,
        stream_id: str,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Get state of a stream."""
        state = self._stream_states.get(stream_id)
        if not state:
            return {"error": "Stream not found"}
        
        return state.to_dict()
    
    async def list_streams(self, ctx: Any = None) -> Dict[str, Any]:
        """List all active streams."""
        streams = []
        
        for stream_id, state in self._stream_states.items():
            streams.append({
                "stream_id": stream_id,
                "session_id": state.session_id,
                "status": state.status.value,
                "format": state.format.value,
                "tokens_generated": state.tokens_generated,
            })
        
        return {"streams": streams, "count": len(streams)}
    
    async def get_stats(self, ctx: Any = None) -> Dict[str, Any]:
        """Get streaming metrics."""
        if not self._initialized:
            await self.initialize(ctx)
        
        return {
            "metrics": self._metrics.get_metrics() if self._metrics else {},
            "active_streams": len(self._active_streams),
            "connection_count": self._connection_manager.get_connection_count() if self._connection_manager else 0,
        }
    
    async def reload_config(self, ctx: Any = None) -> Dict[str, Any]:
        """Hot-reload configuration."""
        try:
            self._config = self._load_config()
            self._build_configs()
            return {"status": "reloaded"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self, ctx: Any = None) -> Dict[str, Any]:
        """Graceful shutdown."""
        # Close all active streams
        for stream_id in list(self._active_streams.keys()):
            await self.cancel_stream(stream_id)
        
        self._initialized = False
        
        if self._event_bus:
            await self._event_bus.publish(
                "streaming.shutdown",
                {"module": "streaming_rag"},
            )
        
        logger.info("Streaming RAG adapter shut down")
        return {"status": "shutdown"}
    
    async def health_check(self, ctx: Any = None) -> Dict[str, Any]:
        """Check component health."""
        if not self._initialized:
            return {"module": "streaming_rag", "status": "not_initialized"}
        
        return {
            "module": "streaming_rag",
            "status": "healthy",
            "initialized": self._initialized,
            "active_streams": len(self._active_streams),
            "formats_available": ["sse", "websocket", "jsonl", "text"],
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    async def _resolve_llm_module(self) -> Optional[Any]:
        """Resolve LLM module via ProviderMapper chain."""
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            chain = ProviderMapper.resolve_chain("rag")
            for module_name, provider_name in chain:
                module = self._module_registry.get_module(module_name)
                if not module and hasattr(self._module_registry, "resolve_module"):
                    module = await self._module_registry.resolve_module(module_name)
                if module:
                    logger.debug(f"[STREAMING] LLM resolved: {module_name}")
                    return module
                logger.warning(f"[STREAMING] Module '{module_name}' not available, trying next")
        except Exception as e:
            logger.warning(
                f"[STREAMING] ProviderMapper NOT AVAILABLE - using hardcoded fallback "
                f"'inference_ollama_grok'. Centralized provider config (UBP_ROLES__RAG_PROVIDER) "
                f"is IGNORED for this module. Cause: {e}"
            )
        # Legacy fallback
        module = self._module_registry.get_module("inference_ollama_grok")
        if not module and hasattr(self._module_registry, "resolve_module"):
            module = await self._module_registry.resolve_module("inference_ollama_grok")
        return module

    def _cleanup_stream(self, stream_id: str) -> None:
        """Clean up stream resources."""
        if stream_id in self._active_streams:
            del self._active_streams[stream_id]
        if stream_id in self._stream_states:
            del self._stream_states[stream_id]
