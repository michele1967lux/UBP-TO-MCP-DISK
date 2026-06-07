"""
RAG Qdrant Module - Test Suite

Comprehensive tests for enterprise-grade RAG module.
"""

import asyncio
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_module_dir():
    """Create temporary module directory with config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create config.json
        config = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "timeout": 30
            },
            "collection": {
                "default_name": "test_documents",
                "vector_size": 384,
                "distance": "Cosine"
            },
            "embedding": {
                "provider": "mock",
                "model": "mock-model",
                "dimension": 384,
                "batch_size": 32
            },
            "chunking": {
                "enabled": True,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "split_by": "sentence"
            },
            "retrieval": {
                "default_top_k": 5,
                "score_threshold": 0.7
            },
            "reliability": {
                "max_retries": 3,
                "retry_delay_seconds": 0.1,
                "circuit_breaker_threshold": 5,
                "circuit_breaker_timeout": 60
            }
        }
        
        with open(temp_path / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create manifest.json
        manifest = {
            "name": "rag_qdrant_test",
            "version": "1.0.0",
            "description": "Test RAG module",
            "module_type": "rag",
            "requires_event_bus": False,
            "config_file": "config.json"
        }
        
        with open(temp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        yield temp_path


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "doc_id": "doc1",
            "text": "The quick brown fox jumps over the lazy dog. This is a test document.",
            "metadata": {"source": "test", "category": "animals"}
        },
        {
            "doc_id": "doc2",
            "text": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data.",
            "metadata": {"source": "test", "category": "technology"}
        },
        {
            "doc_id": "doc3",
            "text": "Python is a popular programming language. It is used for web development, data science, and automation.",
            "metadata": {"source": "test", "category": "programming"}
        }
    ]


# ============================================================================
# Chunker Tests
# ============================================================================

class TestChunker:
    """Test text chunking functionality."""
    
    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        from rag_qdrant.chunker import ChunkingManager, ChunkingConfig
        
        config = ChunkingConfig(
            strategy="sentence",
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10
        )
        
        manager = ChunkingManager(config)
        
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        chunks = manager.chunk(text)
        
        assert len(chunks) > 0
        assert all(len(c.text) <= config.max_chunk_size for c in chunks)
        assert all(c.text.strip() for c in chunks)
    
    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        from rag_qdrant.chunker import ChunkingManager, ChunkingConfig
        
        config = ChunkingConfig(
            strategy="fixed",
            chunk_size=50,
            chunk_overlap=10
        )
        
        manager = ChunkingManager(config)
        
        text = "A" * 200
        chunks = manager.chunk(text)
        
        assert len(chunks) > 1
    
    def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        from rag_qdrant.chunker import ChunkingManager, ChunkingConfig
        
        config = ChunkingConfig(
            strategy="paragraph",
            chunk_size=200,
            chunk_overlap=20
        )
        
        manager = ChunkingManager(config)
        
        text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph here."
        chunks = manager.chunk(text)
        
        assert len(chunks) >= 1
    
    def test_chunk_with_overlap_ids(self):
        """Test chunking with document ID tracking."""
        from rag_qdrant.chunker import ChunkingManager, ChunkingConfig
        
        config = ChunkingConfig(chunk_size=100)
        manager = ChunkingManager(config)
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = manager.chunk_with_overlap_ids(text, "test_doc")
        
        for chunk in chunks:
            assert "doc_id" in chunk.metadata
            assert chunk.metadata["doc_id"] == "test_doc"
            assert "chunk_id" in chunk.metadata
    
    def test_empty_text(self):
        """Test handling of empty text."""
        from rag_qdrant.chunker import ChunkingManager
        
        manager = ChunkingManager()
        
        chunks = manager.chunk("")
        assert len(chunks) == 0
        
        chunks = manager.chunk("   ")
        assert len(chunks) == 0


# ============================================================================
# Embeddings Tests
# ============================================================================

class TestEmbeddings:
    """Test embedding generation functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_embedding(self):
        """Test mock embedding provider."""
        from rag_qdrant.embeddings import EmbeddingManager, EmbeddingConfig
        
        config = EmbeddingConfig(
            provider="mock",
            dimension=384
        )
        
        manager = EmbeddingManager(config)
        await manager.initialize()
        
        embedding = await manager.embed("Hello world")
        
        assert len(embedding) == 384
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_embedding(self):
        """Test batch embedding generation."""
        from rag_qdrant.embeddings import EmbeddingManager, EmbeddingConfig
        
        config = EmbeddingConfig(
            provider="mock",
            dimension=384,
            batch_size=2
        )
        
        manager = EmbeddingManager(config)
        await manager.initialize()
        
        texts = ["Hello", "World", "Test"]
        embeddings = await manager.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self):
        """Test embedding cache functionality."""
        from rag_qdrant.embeddings import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        
        # Store embedding
        await cache.set("test", "model", [1.0, 2.0, 3.0])
        
        # Retrieve embedding
        result = await cache.get("test", "model")
        assert result == [1.0, 2.0, 3.0]
        
        # Cache miss
        result = await cache.get("nonexistent", "model")
        assert result is None
        
        # Check stats
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


# ============================================================================
# Client Tests
# ============================================================================

class TestQdrantClient:
    """Test Qdrant client functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker behavior."""
        from rag_qdrant.client import CircuitBreaker, CircuitBreakerConfig, CircuitState
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=1
        )
        
        breaker = CircuitBreaker("test", config)
        
        assert breaker.state == CircuitState.CLOSED
        
        # Record failures
        for _ in range(3):
            await breaker.record_failure(Exception("test"))
        
        assert breaker.state == CircuitState.OPEN
        
        # Cannot execute when open
        assert not await breaker.can_execute()
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry with backoff."""
        from rag_qdrant.client import retry_with_backoff, RetryConfig
        
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1
        )
        
        result = await retry_with_backoff(failing_operation, config)
        
        assert result == "success"
        assert call_count == 3


# ============================================================================
# Operations Tests
# ============================================================================

class TestOperations:
    """Test operation handlers."""
    
    def test_validation(self):
        """Test input validation."""
        from rag_qdrant.operations import OperationValidator, ValidationError
        
        # Valid inputs
        OperationValidator.validate_doc_id("valid_id")
        OperationValidator.validate_text("Valid text content")
        OperationValidator.validate_collection_name("valid_collection")
        OperationValidator.validate_top_k(5)
        OperationValidator.validate_score_threshold(0.7)
        
        # Invalid inputs
        with pytest.raises(ValidationError):
            OperationValidator.validate_doc_id("")
        
        with pytest.raises(ValidationError):
            OperationValidator.validate_text("")
        
        with pytest.raises(ValidationError):
            OperationValidator.validate_top_k(0)
        
        with pytest.raises(ValidationError):
            OperationValidator.validate_score_threshold(1.5)
    
    def test_operation_result(self):
        """Test operation result structure."""
        from rag_qdrant.operations import OperationResult, OperationStatus
        
        result = OperationResult(
            status=OperationStatus.SUCCESS,
            operation="test_operation",
            data={"key": "value"},
            duration_ms=10.5
        )
        
        assert result.success
        assert result.to_dict()["status"] == "success"
        assert result.to_dict()["data"]["key"] == "value"


# ============================================================================
# Events Tests
# ============================================================================

class TestEvents:
    """Test event handling functionality."""
    
    @pytest.mark.asyncio
    async def test_event_dispatch(self):
        """Test event dispatching."""
        from rag_qdrant.events import EventManager, Event, EventHandler, ProcessingResult, EventStatus
        
        class TestHandler(EventHandler):
            @property
            def event_types(self):
                return ["test.event"]
            
            async def handle(self, event):
                return ProcessingResult(
                    event_id=event.metadata.event_id,
                    status=EventStatus.COMPLETED,
                    duration_ms=1.0,
                    result_data={"handled": True}
                )
        
        manager = EventManager()
        manager.register_handler(TestHandler())
        
        event = Event(
            event_type="test.event",
            payload={"data": "test"}
        )
        
        results = await manager.dispatch(event)
        
        assert len(results) == 1
        assert results[0].status == EventStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_event_retry(self):
        """Test event retry logic."""
        from rag_qdrant.events import EventManager, Event, EventHandler, ProcessingResult, EventStatus
        
        attempt_count = 0
        
        class FailingHandler(EventHandler):
            @property
            def event_types(self):
                return ["failing.event"]
            
            async def handle(self, event):
                nonlocal attempt_count
                attempt_count += 1
                return ProcessingResult(
                    event_id=event.metadata.event_id,
                    status=EventStatus.FAILED,
                    duration_ms=1.0,
                    error="Test failure"
                )
        
        manager = EventManager(max_retries=3, retry_delay=0.01)
        manager.register_handler(FailingHandler())
        
        event = Event(
            event_type="failing.event",
            payload={}
        )
        
        await manager.dispatch(event)
        
        assert attempt_count == 3


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete module."""
    
    @pytest.mark.asyncio
    async def test_module_lifecycle(self, temp_module_dir):
        """Test module initialization and shutdown."""
        from rag_qdrant import create_module
        
        module = create_module(temp_module_dir)
        
        await module.initialize()
        
        assert module._initialized
        
        health = await module.health_check()
        assert health["status"] in ["healthy", "degraded"]
        
        await module.shutdown()
        
        assert not module._initialized
    
    @pytest.mark.asyncio
    async def test_document_operations(self, temp_module_dir, sample_documents):
        """Test document add, query, delete operations."""
        from rag_qdrant import create_module
        
        module = create_module(temp_module_dir)
        await module.initialize()
        
        try:
            # Add document
            doc = sample_documents[0]
            result = await module.add_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc["metadata"]
            )
            
            assert result["status"] == "success"
            
            # Query
            query_result = await module.query("fox jumps")
            
            assert "results" in query_result
            
            # Delete
            delete_result = await module.delete_document(doc["doc_id"])
            
            assert delete_result["status"] == "success"
            
        finally:
            await module.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, temp_module_dir, sample_documents):
        """Test batch document operations."""
        from rag_qdrant import create_module
        
        module = create_module(temp_module_dir)
        await module.initialize()
        
        try:
            # Batch add
            result = await module.add_documents_batch(sample_documents)
            
            assert result["data"]["total"] == 3
            
            # Query all
            query_result = await module.query("programming")
            
            assert "results" in query_result
            
        finally:
            await module.shutdown()
    
    @pytest.mark.asyncio
    async def test_collection_operations(self, temp_module_dir):
        """Test collection management operations."""
        from rag_qdrant import create_module
        
        module = create_module(temp_module_dir)
        await module.initialize()
        
        try:
            # Create collection
            result = await module.create_collection(
                collection_name="test_collection",
                vector_size=384,
                distance="Cosine"
            )
            
            assert result["status"] == "success"
            
            # List collections
            list_result = await module.list_collections()
            
            assert "collections" in list_result["data"]
            
            # Delete collection
            delete_result = await module.delete_collection("test_collection")
            
            assert delete_result["status"] == "success"
            
        finally:
            await module.shutdown()
    
    @pytest.mark.asyncio
    async def test_stats_and_metrics(self, temp_module_dir, sample_documents):
        """Test statistics and metrics collection."""
        from rag_qdrant import create_module
        
        module = create_module(temp_module_dir)
        await module.initialize()
        
        try:
            # Perform some operations
            await module.add_document(
                doc_id="stats_test",
                text="Test document for statistics",
                metadata={}
            )
            
            await module.query("test")
            
            # Get stats
            stats = await module.get_stats()
            
            assert "operations" in stats
            assert stats["operations"]["documents_added"] >= 1
            assert stats["operations"]["queries_executed"] >= 1
            
        finally:
            await module.shutdown()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_batch(self, temp_module_dir):
        """Test handling of large document batches."""
        from rag_qdrant import create_module
        
        module = create_module(temp_module_dir)
        await module.initialize()
        
        try:
            # Create large batch
            documents = [
                {
                    "doc_id": f"perf_doc_{i}",
                    "text": f"Performance test document number {i}. " * 10,
                    "metadata": {"batch": "performance_test"}
                }
                for i in range(100)
            ]
            
            result = await module.add_documents_batch(documents)
            
            assert result["data"]["success"] == 100
            
        finally:
            await module.shutdown()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
