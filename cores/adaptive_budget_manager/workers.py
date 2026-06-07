"""
Context Governor Background Workers (v3.7.0)

Implements proactive context preparation workers for intelligent memory management:
- MemoryConsolidator: Compresses old conversation turns periodically
- MetadataEnricher: Extracts entities and keywords for context headers
- PreSummarizer: Pre-computes summaries at multiple compression levels
- ContextAnalyzer: Tracks topics and conversation intent

Workers run as background tasks when enabled via config:
  UBP_CONTEXT_GOVERNOR__WORKERS_ENABLED=true

These workers prepare context asynchronously so that when a request arrives,
compression and enrichment are already available, reducing latency.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CompressionLevel(str, Enum):
    """Pre-computed summary compression levels."""
    NONE = "none"       # 100% - original text
    LIGHT = "light"     # 70% - minor compression
    MEDIUM = "medium"   # 50% - moderate compression
    HEAVY = "heavy"     # 30% - aggressive compression
    MINIMAL = "minimal" # 10% - keywords only


@dataclass
class ConversationSummary:
    """Pre-computed summary at a specific compression level."""
    level: CompressionLevel
    text: str
    token_count: int
    created_at: datetime = field(default_factory=datetime.now)
    original_token_count: int = 0
    
    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 1.0
        return self.token_count / self.original_token_count


@dataclass
class ContextMetadata:
    """Extracted metadata for context header injection."""
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    sentiment: str = "neutral"
    intent: str = "unknown"
    language: str = "it"
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass 
class ConversationState:
    """
    Cached state for a conversation session.
    
    Stores pre-computed summaries at multiple compression levels
    and extracted metadata for fast retrieval.
    """
    session_id: str
    turn_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Pre-computed summaries at different compression levels
    summaries: Dict[CompressionLevel, ConversationSummary] = field(default_factory=dict)
    
    # Extracted metadata
    metadata: Optional[ContextMetadata] = None
    
    # Raw conversation history (for incremental processing)
    raw_turns: List[Dict[str, str]] = field(default_factory=list)
    
    def get_best_summary(self, max_tokens: int) -> Optional[ConversationSummary]:
        """
        Get the best available summary that fits within token budget.
        
        Returns the largest summary that fits, or None if nothing fits.
        """
        for level in [CompressionLevel.NONE, CompressionLevel.LIGHT, 
                      CompressionLevel.MEDIUM, CompressionLevel.HEAVY,
                      CompressionLevel.MINIMAL]:
            if level in self.summaries:
                summary = self.summaries[level]
                if summary.token_count <= max_tokens:
                    return summary
        return None


# =============================================================================
# WORKER INTERFACES
# =============================================================================

class BaseWorker:
    """Base class for background workers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the worker background task."""
        if not self.enabled:
            logger.info(f"[WORKER] {self.__class__.__name__} disabled in config")
            return
        
        if self._running:
            logger.warning(f"[WORKER] {self.__class__.__name__} already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[WORKER] {self.__class__.__name__} started")
    
    async def stop(self):
        """Stop the worker gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[WORKER] {self.__class__.__name__} stopped")
    
    async def _run_loop(self):
        """Main worker loop - override in subclasses."""
        raise NotImplementedError
    
    async def process(self, session_id: str, state: ConversationState) -> ConversationState:
        """Process a conversation state - override in subclasses."""
        raise NotImplementedError


# =============================================================================
# MEMORY CONSOLIDATOR
# =============================================================================

class MemoryConsolidator(BaseWorker):
    """
    Compresses old conversation turns periodically.
    
    Strategy:
    - Keep last N turns as raw (configurable via RAW_BUFFER_SIZE)
    - Compress older turns into a narrative summary
    - Trigger compression when total tokens exceed threshold
    """
    
    def __init__(self, config: Dict[str, Any], llm_callback: Callable = None):
        super().__init__(config)
        self.llm_callback = llm_callback
        self.raw_buffer_size = config.get("raw_buffer_size", 10)
        self.compression_interval = config.get("compression_interval_seconds", 60)
        self.max_tokens_before_compression = config.get("max_tokens_before_compression", 4000)
        
        # Session states cache
        self._states: Dict[str, ConversationState] = {}
    
    async def _run_loop(self):
        """Periodically check and compress old sessions."""
        while self._running:
            try:
                await self._consolidate_all()
                await asyncio.sleep(self.compression_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CONSOLIDATOR] Error in consolidation loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _consolidate_all(self):
        """Check all sessions and compress if needed."""
        for session_id, state in list(self._states.items()):
            # Skip recently active sessions
            if datetime.now() - state.last_activity < timedelta(seconds=30):
                continue
            
            # Check if compression needed
            total_tokens = sum(
                len(turn.get("content", "")) // 4  # Rough estimate
                for turn in state.raw_turns
            )
            
            if total_tokens > self.max_tokens_before_compression:
                logger.info(f"[CONSOLIDATOR] Compressing session {session_id} ({total_tokens} tokens)")
                await self.process(session_id, state)
    
    async def process(self, session_id: str, state: ConversationState) -> ConversationState:
        """Compress old turns into summary."""
        if len(state.raw_turns) <= self.raw_buffer_size:
            return state  # Nothing to compress
        
        # Split into old and recent
        old_turns = state.raw_turns[:-self.raw_buffer_size]
        recent_turns = state.raw_turns[-self.raw_buffer_size:]
        
        if not old_turns:
            return state
        
        # Format old turns for summarization
        old_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in old_turns
        )
        
        if self.llm_callback:
            try:
                summary_text = await self.llm_callback(
                    prompt=f"Riassumi in modo conciso la seguente conversazione, "
                           f"mantenendo solo i punti chiave:\n\n{old_text}",
                    max_tokens=500,
                )
                
                # Create summary at MEDIUM level
                summary = ConversationSummary(
                    level=CompressionLevel.MEDIUM,
                    text=summary_text,
                    token_count=len(summary_text) // 4,
                    original_token_count=len(old_text) // 4,
                )
                state.summaries[CompressionLevel.MEDIUM] = summary
                
                # Keep only recent turns
                state.raw_turns = recent_turns
                
                logger.info(
                    f"[CONSOLIDATOR] Session {session_id} compressed: "
                    f"{len(old_turns)} turns → {summary.token_count} tokens"
                )
                
            except Exception as e:
                logger.error(f"[CONSOLIDATOR] LLM summarization failed: {e}")
        
        return state
    
    def register_session(self, session_id: str, state: ConversationState):
        """Register a session for background consolidation."""
        self._states[session_id] = state
    
    def unregister_session(self, session_id: str):
        """Remove session from consolidation tracking."""
        self._states.pop(session_id, None)


# =============================================================================
# METADATA ENRICHER
# =============================================================================

class MetadataEnricher(BaseWorker):
    """
    Extracts entities and keywords from conversation for context headers.
    
    Creates a "Context Header" that can be injected at the start of prompts:
    ```
    [Contesto: Entità: Mario Rossi, Progetto Alpha. Argomenti: budget, timeline]
    ```
    """
    
    def __init__(self, config: Dict[str, Any], llm_callback: Callable = None):
        super().__init__(config)
        self.llm_callback = llm_callback
        self.enrichment_interval = config.get("enrichment_interval_seconds", 30)
        
        self._states: Dict[str, ConversationState] = {}
    
    async def _run_loop(self):
        """Periodically enrich session metadata."""
        while self._running:
            try:
                await self._enrich_all()
                await asyncio.sleep(self.enrichment_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ENRICHER] Error in enrichment loop: {e}")
                await asyncio.sleep(5)
    
    async def _enrich_all(self):
        """Enrich metadata for all active sessions."""
        for session_id, state in list(self._states.items()):
            # Only enrich if new turns since last enrichment
            if state.metadata and state.metadata.extracted_at > state.last_activity:
                continue
            
            await self.process(session_id, state)
    
    async def process(self, session_id: str, state: ConversationState) -> ConversationState:
        """Extract metadata from conversation."""
        if not state.raw_turns:
            return state
        
        # Format recent conversation
        recent_text = "\n".join(
            turn.get("content", "") for turn in state.raw_turns[-5:]
        )
        
        if self.llm_callback:
            try:
                # Extract entities and topics via LLM
                extraction_prompt = (
                    "Analizza il seguente testo ed estrai in JSON:\n"
                    '{"entities": ["lista nomi/organizzazioni"], '
                    '"topics": ["lista argomenti principali"], '
                    '"intent": "domanda|richiesta|informazione|altro"}\n\n'
                    f"Testo:\n{recent_text}\n\nJSON:"
                )
                
                result = await self.llm_callback(
                    prompt=extraction_prompt,
                    max_tokens=200,
                )
                
                # Parse JSON (basic extraction)
                import json
                try:
                    parsed = json.loads(result)
                    state.metadata = ContextMetadata(
                        entities=parsed.get("entities", []),
                        topics=parsed.get("topics", []),
                        intent=parsed.get("intent", "unknown"),
                    )
                    logger.debug(f"[ENRICHER] Session {session_id} enriched: {state.metadata}")
                except json.JSONDecodeError:
                    logger.warning(f"[ENRICHER] Failed to parse metadata JSON: {result}")
                    
            except Exception as e:
                logger.error(f"[ENRICHER] LLM extraction failed: {e}")
        
        return state
    
    def get_context_header(self, session_id: str) -> str:
        """
        Get formatted context header for a session.
        
        Returns string like:
        "[Contesto: Entità: X, Y. Argomenti: A, B. Intent: domanda]"
        """
        state = self._states.get(session_id)
        if not state or not state.metadata:
            return ""
        
        meta = state.metadata
        parts = []
        
        if meta.entities:
            parts.append(f"Entità: {', '.join(meta.entities[:3])}")
        if meta.topics:
            parts.append(f"Argomenti: {', '.join(meta.topics[:3])}")
        if meta.intent != "unknown":
            parts.append(f"Intent: {meta.intent}")
        
        if not parts:
            return ""
        
        return f"[Contesto: {'. '.join(parts)}]"
    
    def register_session(self, session_id: str, state: ConversationState):
        self._states[session_id] = state
    
    def unregister_session(self, session_id: str):
        self._states.pop(session_id, None)


# =============================================================================
# PRE-SUMMARIZER
# =============================================================================

class PreSummarizer(BaseWorker):
    """
    Pre-computes summaries at multiple compression levels.
    
    Maintains summaries at:
    - LIGHT (70%): Minor compression for comfort zone
    - MEDIUM (50%): Moderate compression for tight situations
    - HEAVY (30%): Aggressive compression for critical situations
    - MINIMAL (10%): Keywords only for emergency
    
    When a request arrives, the appropriate summary is instantly available.
    """
    
    def __init__(self, config: Dict[str, Any], llm_callback: Callable = None):
        super().__init__(config)
        self.llm_callback = llm_callback
        self.summarization_interval = config.get("summarization_interval_seconds", 60)
        
        self._states: Dict[str, ConversationState] = {}
        
        # Target compression ratios
        self.compression_targets = {
            CompressionLevel.LIGHT: 0.7,
            CompressionLevel.MEDIUM: 0.5,
            CompressionLevel.HEAVY: 0.3,
            CompressionLevel.MINIMAL: 0.1,
        }
    
    async def _run_loop(self):
        """Periodically pre-compute summaries."""
        while self._running:
            try:
                await self._summarize_all()
                await asyncio.sleep(self.summarization_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PRESUMMARIZER] Error in summarization loop: {e}")
                await asyncio.sleep(5)
    
    async def _summarize_all(self):
        """Generate summaries for all sessions needing updates."""
        for session_id, state in list(self._states.items()):
            # Check if summaries are stale
            for level in [CompressionLevel.LIGHT, CompressionLevel.MEDIUM, 
                          CompressionLevel.HEAVY, CompressionLevel.MINIMAL]:
                if level not in state.summaries:
                    await self._generate_summary(session_id, state, level)
                elif (datetime.now() - state.summaries[level].created_at).seconds > 300:
                    # Refresh if older than 5 minutes
                    await self._generate_summary(session_id, state, level)
    
    async def _generate_summary(
        self, 
        session_id: str, 
        state: ConversationState, 
        level: CompressionLevel
    ):
        """Generate summary at specific compression level."""
        if not state.raw_turns:
            return
        
        # Get full conversation text
        full_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in state.raw_turns
        )
        
        original_tokens = len(full_text) // 4
        target_tokens = int(original_tokens * self.compression_targets[level])
        
        if not self.llm_callback:
            return
        
        try:
            # Generate compression prompt based on level
            if level == CompressionLevel.MINIMAL:
                prompt = (
                    f"Estrai SOLO le parole chiave principali (max 10) "
                    f"dalla seguente conversazione:\n\n{full_text}\n\nParole chiave:"
                )
            elif level == CompressionLevel.HEAVY:
                prompt = (
                    f"Riassumi in modo MOLTO conciso (max {target_tokens * 4} caratteri) "
                    f"la seguente conversazione, mantenendo solo i punti essenziali:\n\n"
                    f"{full_text}\n\nRiassunto brevissimo:"
                )
            elif level == CompressionLevel.MEDIUM:
                prompt = (
                    f"Riassumi in modo conciso (max {target_tokens * 4} caratteri) "
                    f"la seguente conversazione:\n\n{full_text}\n\nRiassunto:"
                )
            else:  # LIGHT
                prompt = (
                    f"Comprimi leggermente (max {target_tokens * 4} caratteri) "
                    f"la seguente conversazione, mantenendo i dettagli importanti:\n\n"
                    f"{full_text}\n\nVersione compressa:"
                )
            
            summary_text = await self.llm_callback(
                prompt=prompt,
                max_tokens=target_tokens,
            )
            
            state.summaries[level] = ConversationSummary(
                level=level,
                text=summary_text,
                token_count=len(summary_text) // 4,
                original_token_count=original_tokens,
            )
            
            logger.debug(
                f"[PRESUMMARIZER] Session {session_id} summary at {level.value}: "
                f"{state.summaries[level].token_count} tokens "
                f"(ratio: {state.summaries[level].compression_ratio:.1%})"
            )
            
        except Exception as e:
            logger.error(f"[PRESUMMARIZER] Failed to generate {level.value} summary: {e}")
    
    def register_session(self, session_id: str, state: ConversationState):
        self._states[session_id] = state
    
    def unregister_session(self, session_id: str):
        self._states.pop(session_id, None)


# =============================================================================
# CONTEXT GOVERNOR WORKERS ORCHESTRATOR
# =============================================================================

class ContextGovernorWorkers:
    """
    Orchestrates all Context Governor background workers.
    
    Usage:
    ```python
    workers = ContextGovernorWorkers(config, llm_callback)
    await workers.start()
    
    # Register session for background processing
    workers.register_session("session123", state)
    
    # Get pre-computed data
    header = workers.get_context_header("session123")
    summary = workers.get_best_summary("session123", max_tokens=1000)
    
    await workers.stop()
    ```
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        llm_callback: Callable = None,
        di_container = None
    ):
        self.config = config.get("context_governor", {})
        self.enabled = self.config.get("workers_enabled", False)
        self.max_workers = self.config.get("max_workers", 10)
        self.llm_callback = llm_callback
        self.di_container = di_container
        
        # Initialize workers
        self.consolidator = MemoryConsolidator(
            config={
                "enabled": self.enabled and self.config.get("consolidator_enabled", True),
                "raw_buffer_size": config.get("raw_buffer_size", 10),
                "compression_interval_seconds": 60,
                "max_tokens_before_compression": config.get("max_context_tokens", 4000),
            },
            llm_callback=llm_callback,
        )
        
        self.enricher = MetadataEnricher(
            config={
                "enabled": self.enabled and self.config.get("enricher_enabled", True),
                "enrichment_interval_seconds": 30,
            },
            llm_callback=llm_callback,
        )
        
        self.presummarizer = PreSummarizer(
            config={
                "enabled": self.enabled and self.config.get("pre_summarize", True),
                "summarization_interval_seconds": 60,
            },
            llm_callback=llm_callback,
        )
        
        logger.info(
            f"[CONTEXT-GOVERNOR-WORKERS] Initialized (enabled={self.enabled})",
            extra={
                "consolidator": self.consolidator.enabled,
                "enricher": self.enricher.enabled,
                "presummarizer": self.presummarizer.enabled,
            }
        )
    
    async def start(self):
        """Start all enabled workers."""
        if not self.enabled:
            logger.info("[CONTEXT-GOVERNOR-WORKERS] Workers disabled, skipping start")
            return
        
        await asyncio.gather(
            self.consolidator.start(),
            self.enricher.start(),
            self.presummarizer.start(),
        )
        logger.info("[CONTEXT-GOVERNOR-WORKERS] All workers started")
    
    async def stop(self):
        """Stop all workers gracefully."""
        await asyncio.gather(
            self.consolidator.stop(),
            self.enricher.stop(),
            self.presummarizer.stop(),
        )
        logger.info("[CONTEXT-GOVERNOR-WORKERS] All workers stopped")
    
    def register_session(self, session_id: str, initial_turns: List[Dict[str, str]] = None):
        """
        Register a conversation session for background processing.
        
        Args:
            session_id: Unique session identifier
            initial_turns: Optional initial conversation turns
        """
        state = ConversationState(
            session_id=session_id,
            raw_turns=initial_turns or [],
        )
        
        self.consolidator.register_session(session_id, state)
        self.enricher.register_session(session_id, state)
        self.presummarizer.register_session(session_id, state)
        
        logger.debug(f"[CONTEXT-GOVERNOR-WORKERS] Session {session_id} registered")
    
    def unregister_session(self, session_id: str):
        """Remove session from all workers."""
        self.consolidator.unregister_session(session_id)
        self.enricher.unregister_session(session_id)
        self.presummarizer.unregister_session(session_id)
        
        logger.debug(f"[CONTEXT-GOVERNOR-WORKERS] Session {session_id} unregistered")
    
    def add_turn(self, session_id: str, role: str, content: str):
        """Add a conversation turn to a session."""
        # Update in all workers
        for worker in [self.consolidator, self.enricher, self.presummarizer]:
            if session_id in worker._states:
                worker._states[session_id].raw_turns.append({
                    "role": role,
                    "content": content,
                })
                worker._states[session_id].turn_count += 1
                worker._states[session_id].last_activity = datetime.now()
    
    def get_context_header(self, session_id: str) -> str:
        """Get formatted context header for injection."""
        return self.enricher.get_context_header(session_id)
    
    def get_best_summary(self, session_id: str, max_tokens: int) -> Optional[str]:
        """
        Get the best pre-computed summary that fits within token budget.
        
        Returns None if no suitable summary is available.
        """
        if session_id not in self.presummarizer._states:
            return None
        
        state = self.presummarizer._states[session_id]
        summary = state.get_best_summary(max_tokens)
        
        return summary.text if summary else None
    
    def get_session_state(self, session_id: str) -> Optional[ConversationState]:
        """Get full session state for debugging/inspection."""
        return self.consolidator._states.get(session_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "CompressionLevel",
    "ConversationSummary",
    "ContextMetadata",
    "ConversationState",
    
    # Workers
    "BaseWorker",
    "MemoryConsolidator",
    "MetadataEnricher",
    "PreSummarizer",
    
    # Orchestrator
    "ContextGovernorWorkers",
]
