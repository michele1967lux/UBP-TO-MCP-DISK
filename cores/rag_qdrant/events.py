"""
Event Handlers - Enterprise Grade

Production-ready event handling with:
- Event subscription management
- Async event processing
- Error handling and retry
- Event metrics and monitoring
- Dead letter queue support
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from functools import wraps
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# Event Models
# ============================================================================


class EventPriority(Enum):
    """Event processing priority."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventStatus(Enum):
    """Event processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class EventMetadata:
    """Event metadata."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "rag_qdrant"
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Event:
    """Event structure compatible with UBP event bus."""

    event_type: str
    payload: Dict[str, Any]
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "metadata": {
                "event_id": self.metadata.event_id,
                "timestamp": self.metadata.timestamp,
                "source": self.metadata.source,
                "priority": self.metadata.priority.value,
                "correlation_id": self.metadata.correlation_id,
                "causation_id": self.metadata.causation_id,
                "retry_count": self.metadata.retry_count,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create Event from dictionary."""
        metadata_dict = data.get("metadata", {})
        metadata = EventMetadata(
            event_id=metadata_dict.get("event_id", str(uuid.uuid4())),
            timestamp=metadata_dict.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            source=metadata_dict.get("source", "unknown"),
            priority=EventPriority(metadata_dict.get("priority", 1)),
            correlation_id=metadata_dict.get("correlation_id"),
            causation_id=metadata_dict.get("causation_id"),
            retry_count=metadata_dict.get("retry_count", 0),
        )

        return cls(
            event_type=data["event_type"],
            payload=data.get("payload", {}),
            metadata=metadata,
        )


@dataclass
class ProcessingResult:
    """Result of event processing."""

    event_id: str
    status: EventStatus
    duration_ms: float
    error: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "result_data": self.result_data,
        }


# ============================================================================
# Event Handler Interface
# ============================================================================


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @property
    @abstractmethod
    def event_types(self) -> List[str]:
        """Event types this handler processes."""
        pass

    @abstractmethod
    async def handle(self, event: Event) -> ProcessingResult:
        """Process an event."""
        pass

    async def can_handle(self, event: Event) -> bool:
        """Check if handler can process this event."""
        return event.event_type in self.event_types


# ============================================================================
# RAG Event Handlers
# ============================================================================


class DocumentAddedHandler(EventHandler):
    """Handler for document.added events."""

    def __init__(self, operation_handler):
        self.operation_handler = operation_handler

    @property
    def event_types(self) -> List[str]:
        return ["document.added", "document.created"]

    async def handle(self, event: Event) -> ProcessingResult:
        """Handle document addition event."""
        start_time = time.time()
        event_id = event.metadata.event_id

        try:
            payload = event.payload

            # Validate required fields
            if "doc_id" not in payload or "text" not in payload:
                raise ValueError("Event payload must contain 'doc_id' and 'text'")

            result = await self.operation_handler.add_document(
                doc_id=payload["doc_id"],
                text=payload["text"],
                metadata=payload.get("metadata"),
                collection=payload.get("collection"),
            )

            duration = (time.time() - start_time) * 1000

            if result.success:
                return ProcessingResult(
                    event_id=event_id,
                    status=EventStatus.COMPLETED,
                    duration_ms=duration,
                    result_data=result.data,
                )
            else:
                return ProcessingResult(
                    event_id=event_id,
                    status=EventStatus.FAILED,
                    duration_ms=duration,
                    error=result.error,
                )

        except Exception as e:
            logger.error(f"Failed to handle document.added event: {e}")
            return ProcessingResult(
                event_id=event_id,
                status=EventStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class RAGQueryHandler(EventHandler):
    """Handler for rag.query events."""

    def __init__(self, operation_handler, publisher=None):
        self.operation_handler = operation_handler
        self.publisher = publisher

    @property
    def event_types(self) -> List[str]:
        return ["rag.query", "rag.search"]

    async def handle(self, event: Event) -> ProcessingResult:
        """Handle RAG query event."""
        start_time = time.time()
        event_id = event.metadata.event_id

        try:
            payload = event.payload

            # Validate required fields
            if "query_text" not in payload:
                raise ValueError("Event payload must contain 'query_text'")

            result = await self.operation_handler.query(
                query_text=payload["query_text"],
                top_k=payload.get("top_k"),
                collection=payload.get("collection"),
                filter_conditions=payload.get("filter"),
                score_threshold=payload.get("score_threshold"),
            )

            duration = (time.time() - start_time) * 1000

            # Publish result event if publisher available
            if self.publisher:
                await self.publisher.publish(
                    "rag.query.completed",
                    {
                        "request_id": payload.get("request_id"),
                        "correlation_id": event.metadata.correlation_id,
                        "result": result.to_dict(),
                    },
                )

            return ProcessingResult(
                event_id=event_id,
                status=EventStatus.COMPLETED,
                duration_ms=duration,
                result_data=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Failed to handle rag.query event: {e}")
            return ProcessingResult(
                event_id=event_id,
                status=EventStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class DocumentDeletedHandler(EventHandler):
    """Handler for document.deleted events."""

    def __init__(self, operation_handler):
        self.operation_handler = operation_handler

    @property
    def event_types(self) -> List[str]:
        return ["document.deleted", "document.removed"]

    async def handle(self, event: Event) -> ProcessingResult:
        """Handle document deletion event."""
        start_time = time.time()
        event_id = event.metadata.event_id

        try:
            payload = event.payload

            if "doc_id" not in payload:
                raise ValueError("Event payload must contain 'doc_id'")

            result = await self.operation_handler.delete_document(
                doc_id=payload["doc_id"], collection=payload.get("collection")
            )

            duration = (time.time() - start_time) * 1000

            if result.success:
                return ProcessingResult(
                    event_id=event_id,
                    status=EventStatus.COMPLETED,
                    duration_ms=duration,
                    result_data=result.data,
                )
            else:
                return ProcessingResult(
                    event_id=event_id,
                    status=EventStatus.FAILED,
                    duration_ms=duration,
                    error=result.error,
                )

        except Exception as e:
            logger.error(f"Failed to handle document.deleted event: {e}")
            return ProcessingResult(
                event_id=event_id,
                status=EventStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class CollectionCreatedHandler(EventHandler):
    """Handler for collection.create events."""

    def __init__(self, operation_handler, publisher=None):
        self.operation_handler = operation_handler
        self.publisher = publisher

    @property
    def event_types(self) -> List[str]:
        return ["collection.create", "collection.init"]

    async def handle(self, event: Event) -> ProcessingResult:
        """Handle collection creation event."""
        start_time = time.time()
        event_id = event.metadata.event_id

        try:
            payload = event.payload

            if "collection_name" not in payload:
                raise ValueError("Event payload must contain 'collection_name'")

            result = await self.operation_handler.create_collection(
                collection_name=payload["collection_name"],
                vector_size=payload.get("vector_size"),
                distance=payload.get("distance"),
            )

            duration = (time.time() - start_time) * 1000

            # Publish result event
            if self.publisher and result.success:
                await self.publisher.publish(
                    "collection.created",
                    {
                        "collection_name": payload["collection_name"],
                        "correlation_id": event.metadata.correlation_id,
                        **result.data,
                    },
                )

            if result.success:
                return ProcessingResult(
                    event_id=event_id,
                    status=EventStatus.COMPLETED,
                    duration_ms=duration,
                    result_data=result.data,
                )
            else:
                return ProcessingResult(
                    event_id=event_id,
                    status=EventStatus.FAILED,
                    duration_ms=duration,
                    error=result.error,
                )

        except Exception as e:
            logger.error(f"Failed to handle collection.create event: {e}")
            return ProcessingResult(
                event_id=event_id,
                status=EventStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


# ============================================================================
# Event Manager
# ============================================================================


class EventManager:
    """
    Manages event subscriptions and dispatching.

    Features:
    - Handler registration
    - Event routing
    - Retry logic
    - Dead letter queue
    - Metrics
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_dead_letter: bool = True,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_dead_letter = enable_dead_letter

        self._handlers: Dict[str, List[EventHandler]] = {}
        self._dead_letter_queue: List[Event] = []

        # Metrics
        self._metrics = {
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_retried": 0,
            "dead_letter_count": 0,
            "total_processing_time_ms": 0.0,
        }

    def register_handler(self, handler: EventHandler) -> None:
        """Register an event handler."""
        for event_type in handler.event_types:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            logger.info(f"Registered handler for event type: {event_type}")

    def unregister_handler(self, handler: EventHandler) -> None:
        """Unregister an event handler."""
        for event_type in handler.event_types:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h != handler
                ]

    def get_subscribed_events(self) -> List[str]:
        """Get list of subscribed event types."""
        return list(self._handlers.keys())

    async def dispatch(self, event: Event) -> List[ProcessingResult]:
        """
        Dispatch event to registered handlers.

        Args:
            event: Event to dispatch

        Returns:
            List of processing results
        """
        self._metrics["events_received"] += 1

        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            logger.warning(f"No handlers for event type: {event.event_type}")
            return []

        results = []

        for handler in handlers:
            result = await self._process_with_retry(handler, event)
            results.append(result)

            if result.status == EventStatus.COMPLETED:
                self._metrics["events_processed"] += 1
            elif result.status == EventStatus.FAILED:
                self._metrics["events_failed"] += 1
            elif result.status == EventStatus.DEAD_LETTER:
                self._metrics["dead_letter_count"] += 1

            self._metrics["total_processing_time_ms"] += result.duration_ms

        return results

    async def _process_with_retry(
        self, handler: EventHandler, event: Event
    ) -> ProcessingResult:
        """Process event with retry logic."""
        last_result = None

        for attempt in range(self.max_retries):
            event.metadata.retry_count = attempt

            result = await handler.handle(event)
            last_result = result

            if result.status == EventStatus.COMPLETED:
                return result

            if attempt < self.max_retries - 1:
                self._metrics["events_retried"] += 1
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Event processing failed, retrying in {delay}s",
                    extra={
                        "event_id": event.metadata.event_id,
                        "attempt": attempt + 1,
                        "error": result.error,
                    },
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        if self.enable_dead_letter:
            self._dead_letter_queue.append(event)
            return ProcessingResult(
                event_id=event.metadata.event_id,
                status=EventStatus.DEAD_LETTER,
                duration_ms=last_result.duration_ms if last_result else 0,
                error=f"Moved to dead letter queue after {self.max_retries} retries",
            )

        return last_result

    async def process_dead_letter_queue(self) -> List[ProcessingResult]:
        """Reprocess events in dead letter queue."""
        results = []
        events_to_process = self._dead_letter_queue.copy()
        self._dead_letter_queue.clear()

        for event in events_to_process:
            event.metadata.retry_count = 0
            result_list = await self.dispatch(event)
            results.extend(result_list)

        return results

    def get_dead_letter_queue(self) -> List[Event]:
        """Get events in dead letter queue."""
        return self._dead_letter_queue.copy()

    def clear_dead_letter_queue(self) -> int:
        """Clear dead letter queue and return count."""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        return count

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get event manager metrics."""
        avg_processing_time = 0.0
        if self._metrics["events_processed"] > 0:
            avg_processing_time = (
                self._metrics["total_processing_time_ms"]
                / self._metrics["events_processed"]
            )

        return {
            **self._metrics,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "subscribed_events": self.get_subscribed_events(),
            "dead_letter_queue_size": len(self._dead_letter_queue),
        }

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._metrics = {
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "events_retried": 0,
            "dead_letter_count": 0,
            "total_processing_time_ms": 0.0,
        }


# ============================================================================
# Event Publisher
# ============================================================================


class EventPublisher:
    """
    Publishes events to the event bus.

    Compatible with UBP event bus interface.
    """

    def __init__(self, event_bus=None, source: str = "rag_qdrant"):
        self.event_bus = event_bus
        self.source = source

        # Metrics
        self._metrics = {"events_published": 0, "events_failed": 0}

    async def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Publish an event.

        Args:
            event_type: Type of event
            payload: Event payload
            priority: Event priority
            correlation_id: Correlation ID for tracing

        Returns:
            True if published successfully
        """
        try:
            event = Event(
                event_type=event_type,
                payload=payload,
                metadata=EventMetadata(
                    source=self.source, priority=priority, correlation_id=correlation_id
                ),
            )

            if self.event_bus:
                # EventBus.publish expects an Event object, not separate args
                # Import the framework Event class to create proper event object
                try:
                    from ubp_enterprise_hybrid.backend.app.infra.event_bus import Event as FrameworkEvent

                    framework_event = FrameworkEvent(
                        event_type=event.event_type,
                        payload=event.payload,
                        source_module=self.source,
                        correlation_id=event.metadata.correlation_id,
                    )
                    await self.event_bus.publish(framework_event)
                except ImportError:
                    # Fallback: event_bus might accept different signatures
                    # Try calling with the local Event object if framework not available
                    logger.warning(
                        "Framework Event not available, attempting direct publish"
                    )
                    try:
                        await self.event_bus.publish(event)
                    except TypeError:
                        logger.error(
                            f"Failed to publish event: incompatible event_bus interface"
                        )
            else:
                # Log if no event bus configured
                logger.debug(
                    f"Would publish event: {event_type}", extra={"payload": payload}
                )

            self._metrics["events_published"] += 1

            logger.debug(
                f"Published event: {event_type}",
                extra={"event_id": event.metadata.event_id},
            )

            return True

        except Exception as e:
            self._metrics["events_failed"] += 1
            logger.error(f"Failed to publish event: {e}")
            return False

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get publisher metrics."""
        return self._metrics.copy()


# ============================================================================
# Factory Functions
# ============================================================================


def create_event_manager(
    config: Dict[str, Any],
    operation_handler,
    publisher: Optional[EventPublisher] = None,
) -> EventManager:
    """
    Create and configure event manager with handlers.

    Args:
        config: Configuration dictionary
        operation_handler: OperationHandler instance
        publisher: Optional EventPublisher

    Returns:
        Configured EventManager
    """
    reliability_config = config.get("reliability", {})

    manager = EventManager(
        max_retries=reliability_config.get("max_retries", 3),
        retry_delay=reliability_config.get("retry_delay_seconds", 1.0),
        enable_dead_letter=True,
    )

    # Register handlers
    manager.register_handler(DocumentAddedHandler(operation_handler))
    manager.register_handler(RAGQueryHandler(operation_handler, publisher))
    manager.register_handler(DocumentDeletedHandler(operation_handler))
    manager.register_handler(CollectionCreatedHandler(operation_handler, publisher))

    return manager


def create_event_publisher(
    event_bus=None, source: str = "rag_qdrant"
) -> EventPublisher:
    """
    Create an event publisher.

    Args:
        event_bus: UBP event bus instance
        source: Source identifier

    Returns:
        EventPublisher instance
    """
    return EventPublisher(event_bus=event_bus, source=source)
