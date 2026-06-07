# Streaming RAG

**Streaming Response Generation Engine** for real-time RAG responses

Version: 1.0.0 | Architecture: 3-file-pattern | Module Type: core

---

## Overview

`streaming_rag` provides real-time streaming capabilities for RAG pipelines:

| Format | Protocol | Use Case |
|--------|----------|----------|
| **SSE** | Server-Sent Events | Web browsers, HTTP streaming |
| **WebSocket** | Bidirectional | Real-time apps, chat interfaces |
| **JSONL** | JSON Lines | CLI tools, log processing |
| **Text** | Plain text | Simple integrations |
| **Generator** | Python async | Internal module use |

---

## Architecture

```
streaming_rag/
├── __init__.py          # Module factory for ModuleLoader
├── adapter.py           # Bridge layer - exposes 15+ operations
├── providers.py         # Core: Events, State, Buffer, Metrics
├── streams.py           # Stream implementations (SSE, WS, etc.)
├── handlers.py          # Protocol handlers, Response builder
├── config.json          # 150+ environment variables
├── manifest.json        # Operation definitions
└── README.md            # This file
```

---

## Streaming Flow

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                  Streaming Pipeline                      │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  Retrieval   │───▶│  Generation  │───▶│   Done    │  │
│  │    Phase     │    │    Phase     │    │   Event   │  │
│  └──────┬───────┘    └──────┬───────┘    └───────────┘  │
│         │                   │                            │
│         ▼                   ▼                            │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ Source Events│    │ Token Events │                   │
│  └──────────────┘    └──────────────┘                   │
│                                                          │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  Stream   │
                    │  Output   │
                    │(SSE/WS/..)│
                    └───────────┘
```

---

## Event Types

### Lifecycle Events

| Event | Description |
|-------|-------------|
| `stream_start` | Stream initiated |
| `stream_end` | Stream completed |
| `stream_error` | Error occurred |
| `done` | Successfully finished |

### Retrieval Events

| Event | Description |
|-------|-------------|
| `retrieval_start` | Retrieval phase started |
| `retrieval_progress` | Progress update |
| `retrieval_complete` | Retrieval finished |
| `source_found` | Document found |

### Generation Events

| Event | Description |
|-------|-------------|
| `generation_start` | LLM generation started |
| `token` | Single token generated |
| `chunk` | Multiple tokens as chunk |
| `generation_progress` | Progress update |
| `generation_complete` | Generation finished |

### Metadata Events

| Event | Description |
|-------|-------------|
| `metadata` | Custom metadata |
| `timing` | Performance metrics |
| `heartbeat` | SSE keepalive |

---

## Usage Examples

### SSE Streaming (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_query(query: str):
    async def generate():
        async for chunk in streaming.stream_sse(query=query):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
```

### Client-Side (JavaScript)

```javascript
const eventSource = new EventSource('/stream?query=What is RAG?');

eventSource.addEventListener('rag_token', (event) => {
    const data = JSON.parse(event.data);
    document.getElementById('output').innerHTML += data.token;
});

eventSource.addEventListener('rag_source_found', (event) => {
    const data = JSON.parse(event.data);
    console.log('Source:', data.title, data.relevance_score);
});

eventSource.addEventListener('rag_done', (event) => {
    eventSource.close();
});
```

### WebSocket Streaming

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    query = await websocket.receive_text()
    
    await streaming.stream_websocket(
        query=query,
        websocket=websocket,
    )
```

### JSON Lines Streaming

```python
async def stream_jsonl(query: str):
    async for line in streaming.stream_jsonl(query=query):
        # Each line is a JSON object followed by newline
        yield line
```

### Plain Text Streaming

```python
async def stream_text(query: str):
    async for token in streaming.stream_text(query=query):
        # Just the token text, no events
        print(token, end="", flush=True)
```

### Manual Stream Control

```python
# Create stream
result = await streaming.create_stream(
    session_id="user_123",
    query="Explain transformers",
    format="generator",
)
stream_id = result["stream_id"]

# Emit tokens manually
for word in ["Transformers", " are", " neural", " networks"]:
    await streaming.emit_token(stream_id, word, index=i)

# Emit custom event
await streaming.emit_event(
    stream_id,
    event_type="metadata",
    data={"custom_field": "value"},
)

# Cancel if needed
await streaming.cancel_stream(stream_id)
```

---

## SSE Event Format

```
event: rag_token
id: stream_123_1
data: {"event_id":"stream_123_1","event_type":"token","token":"Hello","token_index":0}

event: rag_source_found
id: stream_123_2
data: {"event_id":"stream_123_2","event_type":"source_found","source_id":"doc_1","title":"Introduction to RAG","relevance_score":0.95}

event: rag_done
id: stream_123_50
data: {"event_id":"stream_123_50","event_type":"done","total_tokens":48,"total_duration_ms":1250.5}
```

---

## Buffering & Flow Control

### Token Buffer

```python
# Buffer config
buffer_tokens: 5       # Accumulate N tokens before flush
buffer_timeout_ms: 100 # Flush after timeout even if < N tokens
max_buffer_size: 1000  # Maximum buffer size
overflow_strategy: "drop_oldest"  # or "flush"
```

### Backpressure Handling

```
                    ┌─────────────────────┐
                    │     Event Queue     │
                    │                     │
    ─────────────────▶  [e][e][e][e][e]  ├──────────────▶
                    │                     │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Backpressure Ctrl  │
                    │                     │
                    │  high_watermark: 80 │
                    │  low_watermark: 20  │
                    └─────────────────────┘

When queue > 80% → PAUSE producer
When queue < 20% → RESUME producer
```

### Rate Limiting

```bash
UBP_STREAMING__RATE_LIMIT_ENABLED=true
UBP_STREAMING__TOKENS_PER_SEC=100
UBP_STREAMING__BURST_SIZE=50
```

---

## Metrics

```python
stats = await streaming.get_stats()

# Example output:
{
    "metrics": {
        "total_streams": 500,
        "completed_streams": 485,
        "failed_streams": 15,
        "success_rate": 0.97,
        "total_tokens_generated": 125000,
        "ttft_stats": {
            "avg_ms": 450.5,
            "min_ms": 120.0,
            "max_ms": 2500.0
        },
        "throughput_stats": {
            "avg_tokens_per_sec": 45.2
        },
        "token_latency_stats": {
            "avg_ms": 22.1
        }
    },
    "active_streams": 5,
    "connection_count": 5
}
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time To First Token |
| **TPS** | Tokens Per Second |
| **Inter-token latency** | Time between tokens |
| **Success rate** | Completed / Total |

---

## Configuration

### Key Environment Variables

```bash
# Core
UBP_STREAMING__ENABLED=true
UBP_STREAMING__DEFAULT_FORMAT=sse
UBP_STREAMING__MAX_CONCURRENT=50

# Token Streaming
UBP_STREAMING__CHUNK_SIZE=1
UBP_STREAMING__FLUSH_INTERVAL=50

# Buffering
UBP_STREAMING__BUFFER_TOKENS=5
UBP_STREAMING__BUFFER_TIMEOUT=100
UBP_STREAMING__MAX_BUFFER_SIZE=1000

# SSE
UBP_STREAMING__SSE_PREFIX=rag
UBP_STREAMING__SSE_KEEPALIVE=15000
UBP_STREAMING__SSE_RETRY=3000

# WebSocket
UBP_STREAMING__WS_PING_INTERVAL=30000
UBP_STREAMING__WS_MAX_MSG_SIZE=65536

# Backpressure
UBP_STREAMING__BACKPRESSURE_ENABLED=true
UBP_STREAMING__HIGH_WATERMARK=80
UBP_STREAMING__LOW_WATERMARK=20

# Timeouts
UBP_STREAMING__STREAM_TIMEOUT=300
UBP_STREAMING__IDLE_TIMEOUT=60
UBP_STREAMING__FIRST_TOKEN_TIMEOUT=30

# Events
UBP_STREAMING__EMIT_RETRIEVAL=true
UBP_STREAMING__EMIT_SOURCES=true
UBP_STREAMING__EMIT_PROGRESS=true
UBP_STREAMING__EMIT_TIMING=true
```

---

## Integration

### With retrieval_strategy

```python
# Streaming automatically uses retrieval_strategy
async for event in streaming.stream_query(query):
    if event.event_type == "source_found":
        # Sources from retrieval_strategy
        print(f"Found: {event.title}")
```

### With LLM Module

```python
# Streaming passthrough from LLM
llm_integration:
  passthrough_streaming: true
  llm_module: inference_ollama_grok
  llm_operation: stream_generate
```

### Custom Pipeline

```python
# Create custom streaming pipeline
stream = await streaming.create_stream(session_id, query)
pipeline = StreamingPipeline(stream)

# Run retrieval
sources = await pipeline.stream_retrieval(retrieval_coro)

# Run generation
answer = await pipeline.stream_generation(generation_iter)
```

---

## Error Handling

### Partial Results

```python
# On error, partial content is preserved
{
    "event_type": "stream_error",
    "error_code": "GENERATION_ERROR",
    "error_message": "LLM timeout",
    "recoverable": false,
    "has_partial_result": true
}

# Access partial result
state = await streaming.get_stream_state(stream_id)
partial = state["accumulated_content"]
```

### Graceful Degradation

```bash
UBP_STREAMING__PARTIAL_ON_ERROR=true
UBP_STREAMING__GRACEFUL_DEGRADE=true
```

---

## Operations Reference

### Streaming Operations

| Operation | Description |
|-----------|-------------|
| `stream_query` | Full streaming RAG |
| `stream_sse` | SSE format |
| `stream_websocket` | WebSocket |
| `stream_jsonl` | JSON Lines |
| `stream_text` | Plain text |

### Manual Control

| Operation | Description |
|-----------|-------------|
| `create_stream` | Create stream manually |
| `emit_token` | Emit token to stream |
| `emit_event` | Emit custom event |
| `cancel_stream` | Cancel stream |

### Management

| Operation | Description |
|-----------|-------------|
| `get_stream_state` | Get stream state |
| `list_streams` | List active streams |
| `get_stats` | Get metrics |

---

## Deployment

### 1. Copy Module

```bash
cp -r streaming_rag/ modules/cores/streaming_rag/
```

### 2. Configure

```bash
# .env
UBP_STREAMING__DEFAULT_FORMAT=sse
UBP_STREAMING__MAX_CONCURRENT=100
```

### 3. Initialize

```python
from modules.cores.streaming_rag import create_module

streaming = create_module(
    module_path=Path("modules/cores/streaming_rag"),
    di_container=container,
    event_bus=event_bus,
)

await streaming.initialize()
```

### 4. Expose Endpoint

```python
@app.get("/api/stream")
async def stream_endpoint(query: str):
    return StreamingResponse(
        streaming.stream_sse(query),
        media_type="text/event-stream",
    )
```

---

## Best Practices

### 1. Choose Right Format

| Scenario | Format |
|----------|--------|
| Web browser | SSE |
| Mobile app | WebSocket |
| CLI tool | JSONL or Text |
| Internal | Generator |

### 2. Buffer Tuning

- High latency network → Larger buffer
- Low latency requirement → Smaller buffer
- Word-by-word display → `chunk_size=1`

### 3. Error Handling

```python
try:
    async for event in streaming.stream_query(query):
        if event.event_type == "stream_error":
            handle_error(event)
            break
        process(event)
except Exception as e:
    # Fallback
    pass
```

---

## Troubleshooting

### High TTFT
- Check retrieval timeout
- Reduce document count
- Enable retrieval caching

### Dropped Tokens
- Increase buffer size
- Enable backpressure
- Check network latency

### Connection Timeout
- Increase keepalive interval
- Check SSE retry timeout
- Verify idle timeout settings

---

## Dependencies

### Required
- Python 3.10+
- asyncio

### Optional
- FastAPI/Starlette (SSE responses)
- aiohttp (WebSocket)
- inference_ollama_grok (LLM)
- retrieval_strategy (retrieval)

---

## Changelog

### v1.0.0 (2025-01)
- Initial release
- SSE, WebSocket, JSONL, Text formats
- Token buffering and backpressure
- Progress and timing events
- Source document streaming
- Connection management
- Comprehensive metrics
- Error recovery with partial results
