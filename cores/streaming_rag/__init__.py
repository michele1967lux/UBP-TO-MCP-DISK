"""
streaming_rag - Streaming Response Generation Engine

Core module for UBP Enterprise Hybrid.

Implements real-time streaming for RAG responses:

1. Token Streaming:
   - Token-by-token generation
   - Chunk-based delivery
   - Buffered streaming
   - Rate-limited output

2. Protocol Support:
   - Server-Sent Events (SSE)
   - WebSocket streaming
   - Async generators
   - HTTP chunked transfer

3. Event Types:
   - retrieval_start/complete
   - generation_start/token/complete
   - source_found
   - error
   - done

4. Flow Control:
   - Backpressure handling
   - Client disconnection detection
   - Graceful cancellation
   - Timeout management

5. Integration:
   - Works with all RAG pipelines
   - LLM streaming passthrough
   - Retrieval progress events
   - Unified event format

Features:
- Multiple output formats (SSE, WebSocket, generator)
- Configurable chunk sizes and delays
- Progress callbacks
- Error recovery with partial results
- Cross-module integration
- Production-ready with health checks

v1.0.0: Initial release with full enterprise features

Architecture:
- adapter.py: Bridge layer exposing all operations
- providers.py: Core data classes, buffers, events
- streams.py: Stream implementations
- handlers.py: Protocol-specific handlers
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .adapter import StreamingRAGAdapter

__version__ = "1.0.0"
__all__ = ["create_module", "StreamingRAGAdapter"]


def create_module(
    module_path: Path,
    di_container: Optional[Any] = None,
    event_bus: Optional[Any] = None,
) -> StreamingRAGAdapter:
    """
    Factory function for creating the streaming_rag adapter.
    
    This is the entry point used by ModuleLoader.
    
    Args:
        module_path: Path to the module directory
        di_container: DI container for dependency resolution
        event_bus: Event bus for publishing events
    
    Returns:
        Configured StreamingRAGAdapter instance
    """
    return StreamingRAGAdapter(
        module_path=module_path,
        di_container=di_container,
        event_bus=event_bus,
    )
