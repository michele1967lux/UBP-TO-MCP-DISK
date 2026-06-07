"""
Context Compression Engine — Provider (Pure Business Logic)

Implements the multi-layer evolved context compression system.

Responsibilities:
    - Manage Layer 0 sub-layer lifecycle (create, store, retrieve)
    - Trigger LLM-based compression when threshold is reached
    - Generate Layer 1 (medium-term) and Layer 2 (long-term) via LLM
    - Resolve compression LLM provider via ProviderMapper
    - Comprehensive logging for debug/test

Zero UBP framework dependencies in this file — all framework
integration happens in adapter.py.

v1.0.0 — Initial implementation
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    CompressionEngineConfig,
    CompressionMode,
    CompressionResult,
    CompressionState,
    Layer0State,
    Layer1,
    Layer2,
    ProfileType,
    SubLayer0,
    CompressionProfileSettings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENV-based configuration (shared across both compression modules)
# ---------------------------------------------------------------------------
_COMPRESSION_PROVIDER = os.environ.get("UBP_COMPRESSION__PROVIDER", "")
_COMPRESSION_ROLE = os.environ.get("UBP_COMPRESSION__ROLE", "enrichment")
_COMPRESSION_TEMPERATURE = float(os.environ.get("UBP_COMPRESSION__TEMPERATURE", "0.3"))


# ---------------------------------------------------------------------------
# Prompt templates for LLM compression
# ---------------------------------------------------------------------------

_LAYER0_EXTRACTION_PROMPT = """\
You are a context-analysis assistant.  Given the following assistant response \
and user query, extract a structured JSON sub-layer with these fields:

- "focus": a one-sentence description of the user's current intent
- "user_preferences": a dict of any user preferences expressed or implied
- "query_summary": a condensed version of the user query
- "query_type": one of [question, command, follow_up, clarification, greeting, general]
- "entities_mentioned": list of key entities (names, products, concepts)
- "topic": the main topic
- "sentiment": one of [positive, negative, neutral, mixed]
- "language": detected language code (e.g. "en", "it")
- "response_quality_signals": dict with any quality indicators (e.g. "confidence", "completeness")
- "extra": dict with any other relevant context

USER QUERY:
{query}

ASSISTANT RESPONSE (first 2000 chars):
{response_snippet}

Return ONLY valid JSON (no markdown fences, no explanation)."""

_COMPRESSION_PROMPT = """\
You are an expert memory-compression assistant.  You receive a batch of \
short-term memory sub-layers from a conversation and must produce TWO \
compressed memory layers as valid JSON.

**Layer 1 (medium-term)** — Summarise the conversation window:
  - "conversation_focus": overall theme/focus
  - "user_rules": list of explicit rules/constraints from user
  - "user_specific_requests": list of specific instructions from user
  - "conversation_description": narrative of the conversation flow
  - "key_topics": list of main topics discussed
  - "aggregated_preferences": merged user preferences dict
  - "key_entities": important entities across the window
  - "language": primary language

**Layer 2 (long-term)** — Deep compress for long-term retention:
  - "session_summary": ultra-concise session summary (1-2 sentences)
  - "persistent_user_traits": lasting user characteristics dict
  - "topic_arc": ordered list of major topic transitions
  - "interaction_patterns": observed patterns dict
  - "accumulated_knowledge": key established facts list
  - "unresolved_threads": open questions/topics list

{layer1_context}

SUB-LAYERS TO COMPRESS:
{sub_layers_json}

Return ONLY valid JSON with keys "layer_1" and "layer_2" (no markdown fences).\
"""


# ---------------------------------------------------------------------------
# ContextCompressionProvider
# ---------------------------------------------------------------------------

class ContextCompressionProvider:
    """
    Core provider for multi-layer context compression.

    This class is framework-agnostic. It receives an LLM callable
    (or resolves one lazily via DI) and manages the compression lifecycle.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        di_container: Any = None,
        llm_adapter: Any = None,
    ):
        self._raw_config = config
        self._config = CompressionEngineConfig(**config)
        self._di_container = di_container
        self._llm_provider: Any = llm_adapter
        self._llm_retry_interval: float = 30.0
        self._llm_last_attempt: float = 0.0

        # In-memory session states (standalone — no Redis dependency)
        self._sessions: Dict[str, CompressionState] = {}

        logger.info(
            "[CCE] ContextCompressionProvider initialised | "
            "chat_trigger=%d | agent_trigger=%d | enabled=%s",
            self._config.chat_profile.compression_trigger_threshold,
            self._config.agent_loop_profile.compression_trigger_threshold,
            self._config.enabled,
        )

    # ------------------------------------------------------------------
    # LLM resolution (lazy)
    # ------------------------------------------------------------------

    async def _get_llm(self) -> Any:
        """
        Lazily resolve the LLM provider for compression.

        Resolution strategy (ENV-driven, no hardcoded fallback):
        1. Explicit llm_adapter passed at construction → use directly.
        2. If UBP_COMPRESSION__PROVIDER is set, try that provider first.
        3. Always fall through to resolve_chain(UBP_COMPRESSION__ROLE)
           which returns a health-filtered N-level chain via ProviderMapper.
        """
        if self._llm_provider is not None:
            return self._llm_provider

        now = time.time()
        if now - self._llm_last_attempt < self._llm_retry_interval:
            logger.debug("[CCE] Skipping LLM resolution — retry interval not elapsed")
            return None

        self._llm_last_attempt = now

        if self._di_container is None:
            logger.warning("[CCE] No DI container — cannot resolve LLM provider")
            return None

        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper

            # Build candidate chain: explicit provider + resolve_chain
            candidates: list[tuple[str, str]] = []

            # 1. Explicit provider from ENV, config override, or profile
            explicit = (
                _COMPRESSION_PROVIDER
                or self._config.compression_provider_override
                or ""
            )
            if explicit and explicit in ProviderMapper.PROVIDER_MAP:
                candidates.append(ProviderMapper.PROVIDER_MAP[explicit])

            # 2. Full resolve_chain for role (health-aware, N-level)
            role = _COMPRESSION_ROLE
            chain = ProviderMapper.resolve_chain(role)
            for entry in chain:
                if entry not in candidates:
                    candidates.append(entry)

            # 3. Walk chain, resolve first available module
            for module_name, _provider_name in candidates:
                try:
                    llm_module = await self._di_container.resolve(module_name)
                    if llm_module:
                        self._llm_provider = llm_module
                        logger.info(
                            "[CCE] LLM resolved: %s/%s (role=%s)",
                            module_name, _provider_name, role,
                        )
                        return self._llm_provider
                except Exception as chain_err:
                    logger.warning("[CCE] Chain link failed: %s — %s", module_name, chain_err)
                    continue

            logger.warning(
                "[CCE] No LLM resolved from chain (%d candidates, role=%s)",
                len(candidates), role,
            )
        except Exception as exc:
            logger.error("[CCE] LLM resolution error: %s", exc, exc_info=True)

        return None

    async def _call_llm(
        self,
        prompt: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.3,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        """
        Call the resolved LLM with a prompt and return the raw text response.
        """
        llm = await self._get_llm()
        if llm is None:
            logger.error("[CCE] No LLM available — compression will be skipped")
            return None

        model = model_override or self._config.compression_model_override
        provider = provider_override or self._config.compression_provider_override

        # Only pass model/provider when non-None to let inference module
        # use its own defaults instead of receiving explicit None.
        extra_kwargs: Dict[str, Any] = {}
        if model:
            extra_kwargs["model"] = model
        if provider:
            extra_kwargs["provider"] = provider

        try:
            # Attempt the standard inference interface
            if hasattr(llm, "generate"):
                result = await llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra_kwargs,
                )
                if isinstance(result, dict):
                    return result.get("text") or result.get("response") or result.get("content", "")
                return str(result) if result else None

            if hasattr(llm, "chat"):
                result = await llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra_kwargs,
                )
                if isinstance(result, dict):
                    return result.get("text") or result.get("response") or result.get("content", "")
                return str(result) if result else None

            logger.error("[CCE] LLM module has no generate() or chat() method")
            return None
        except Exception as exc:
            logger.error("[CCE] LLM call failed: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> CompressionState:
        """Get existing session state or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = CompressionState(
                session_id=session_id,
                layer_0=Layer0State(session_id=session_id),
            )
            logger.info("[CCE] New session created: %s", session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[CompressionState]:
        """Get session state if it exists."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete session state. Returns True if session existed."""
        removed = self._sessions.pop(session_id, None)
        if removed:
            logger.info("[CCE] Session deleted: %s", session_id)
        return removed is not None

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Layer 0: Sub-layer creation
    # ------------------------------------------------------------------

    async def create_sub_layer(
        self,
        session_id: str,
        turn_number: int,
        query: str,
        response: str,
        profile_type: ProfileType = ProfileType.CHAT,
        *,
        manual_sub_layer: Optional[Dict[str, Any]] = None,
    ) -> SubLayer0:
        """
        Create a new Layer 0 sub-layer for a given interaction turn.

        If an LLM is available, the sub-layer is extracted from the
        query/response via the extraction prompt. Otherwise a minimal
        sub-layer is constructed from the raw inputs.

        Args:
            session_id: Session identifier
            turn_number: Interaction turn number (0-based)
            query: User query text
            response: Assistant response text
            profile_type: Profile governing compression parameters
            manual_sub_layer: Optional dict to use instead of LLM extraction

        Returns:
            The created SubLayer0 instance
        """
        state = self.get_or_create_session(session_id)
        profile = self._config.get_profile(profile_type)

        logger.info(
            "[CCE] Creating sub-layer | session=%s turn=%d profile=%s",
            session_id, turn_number, profile_type.value,
        )

        if manual_sub_layer:
            sub_layer = SubLayer0(turn_number=turn_number, **manual_sub_layer)
        else:
            sub_layer = await self._extract_sub_layer_via_llm(
                turn_number=turn_number,
                query=query,
                response=response,
            )

        state.layer_0.add_sub_layer(sub_layer)

        logger.info(
            "[CCE] Sub-layer added | session=%s turn=%d sub_layer_id=%s total=%d",
            session_id, turn_number, sub_layer.sub_layer_id, state.layer_0.count,
        )

        return sub_layer

    async def _extract_sub_layer_via_llm(
        self,
        turn_number: int,
        query: str,
        response: str,
    ) -> SubLayer0:
        """Extract sub-layer fields using LLM, with fallback to heuristic."""
        response_snippet = response[:2000] if response else ""
        prompt = _LAYER0_EXTRACTION_PROMPT.format(
            query=query, response_snippet=response_snippet,
        )

        raw = await self._call_llm(prompt, max_tokens=600, temperature=0.2)
        if raw:
            try:
                data = self._parse_json(raw)
                return SubLayer0(
                    turn_number=turn_number,
                    focus=data.get("focus", query[:120]),
                    user_preferences=data.get("user_preferences", {}),
                    query_summary=data.get("query_summary", query[:200]),
                    query_type=data.get("query_type", "general"),
                    entities_mentioned=data.get("entities_mentioned", []),
                    topic=data.get("topic", ""),
                    sentiment=data.get("sentiment", "neutral"),
                    language=data.get("language", "auto"),
                    response_quality_signals=data.get("response_quality_signals", {}),
                    extra=data.get("extra", {}),
                )
            except Exception as parse_err:
                logger.warning(
                    "[CCE] LLM extraction parse failed, using heuristic: %s", parse_err,
                )

        # Heuristic fallback
        logger.debug("[CCE] Using heuristic sub-layer extraction for turn %d", turn_number)
        return SubLayer0(
            turn_number=turn_number,
            focus=query[:120],
            user_preferences={},
            query_summary=query[:200],
            query_type="general",
            entities_mentioned=[],
            topic="",
            sentiment="neutral",
            language="auto",
        )

    # ------------------------------------------------------------------
    # Compression: Layer 0 → Layer 1 + Layer 2
    # ------------------------------------------------------------------

    def should_compress(
        self,
        session_id: str,
        profile_type: ProfileType = ProfileType.CHAT,
    ) -> bool:
        """
        Check whether compression should be triggered for a session.

        Compression triggers when the number of uncompressed sub-layers
        (those added since the last compression) equals or exceeds
        the configured threshold.
        """
        state = self.get_session(session_id)
        if state is None:
            return False

        profile = self._config.get_profile(profile_type)
        threshold = profile.compression_trigger_threshold

        # Count sub-layers since last compression
        uncompressed = [
            sl for sl in state.layer_0.sub_layers
            if sl.turn_number > state.last_compression_turn
        ]
        needs = len(uncompressed) >= threshold
        logger.debug(
            "[CCE] should_compress | session=%s uncompressed=%d threshold=%d → %s",
            session_id, len(uncompressed), threshold, needs,
        )
        return needs

    async def compress(
        self,
        session_id: str,
        profile_type: ProfileType = ProfileType.CHAT,
        *,
        mode: CompressionMode = CompressionMode.THRESHOLD,
        force: bool = False,
    ) -> CompressionResult:
        """
        Execute multi-layer compression on a session.

        Collects uncompressed sub-layers, sends them to the LLM, and
        produces Layer 1 + Layer 2.  The sub-layers are *not* deleted
        from Layer 0 — they remain available for context injection, but
        the ``last_compression_turn`` marker advances.

        Args:
            session_id: Target session
            profile_type: Profile for parameter selection
            mode: How compression was triggered
            force: Bypass the threshold check

        Returns:
            CompressionResult with generated layers
        """
        t0 = time.time()
        state = self.get_or_create_session(session_id)
        profile = self._config.get_profile(profile_type)

        if not force and not self.should_compress(session_id, profile_type):
            logger.info("[CCE] Compression not needed for session=%s", session_id)
            return CompressionResult(success=True, compression_mode=CompressionMode.NONE)

        # Collect sub-layers to compress
        threshold = profile.compression_trigger_threshold
        uncompressed = [
            sl for sl in state.layer_0.sub_layers
            if sl.turn_number > state.last_compression_turn
        ]
        batch = uncompressed[:threshold] if not force else uncompressed

        if not batch:
            return CompressionResult(success=True, compression_mode=CompressionMode.NONE)

        logger.info(
            "[CCE] Compressing %d sub-layers | session=%s mode=%s profile=%s",
            len(batch), session_id, mode.value, profile_type.value,
        )

        # Build compression prompt
        sub_layers_json = json.dumps(
            [sl.model_dump(exclude={"sub_layer_id"}) for sl in batch],
            indent=2, default=str,
        )

        # Include existing Layer 1 context if available
        layer1_context = ""
        if state.layer_1:
            layer1_context = (
                "EXISTING LAYER 1 (update/merge with new information):\n"
                + json.dumps(state.layer_1.model_dump(
                    include={
                        "conversation_focus", "user_rules", "user_specific_requests",
                        "conversation_description", "key_topics", "aggregated_preferences",
                    }
                ), indent=2, default=str)
            )

        prompt = _COMPRESSION_PROMPT.format(
            sub_layers_json=sub_layers_json,
            layer1_context=layer1_context,
        )

        max_tokens = profile.layer1_max_tokens + profile.layer2_max_tokens
        raw = await self._call_llm(prompt, max_tokens=max_tokens, temperature=_COMPRESSION_TEMPERATURE)

        if raw is None:
            logger.error("[CCE] LLM compression call returned None — compression failed")
            return CompressionResult(
                success=False,
                error="LLM unavailable",
                compression_mode=mode,
                latency_ms=(time.time() - t0) * 1000,
            )

        try:
            parsed = self._parse_json(raw)
            l1_data = parsed.get("layer_1", {})
            l2_data = parsed.get("layer_2", {})
        except Exception as parse_err:
            logger.error("[CCE] Compression JSON parse failed: %s", parse_err)
            return CompressionResult(
                success=False,
                error=f"JSON parse error: {parse_err}",
                compression_mode=mode,
                latency_ms=(time.time() - t0) * 1000,
            )

        # Build Layer 1
        turn_range = [batch[0].turn_number, batch[-1].turn_number]
        new_version = (state.layer_1.version + 1) if state.layer_1 else 1
        latency_ms = (time.time() - t0) * 1000

        layer_1 = Layer1(
            session_id=session_id,
            version=new_version,
            source_turn_range=turn_range,
            source_sub_layer_count=len(batch),
            conversation_focus=l1_data.get("conversation_focus", ""),
            user_rules=l1_data.get("user_rules", []),
            user_specific_requests=l1_data.get("user_specific_requests", []),
            conversation_description=l1_data.get("conversation_description", ""),
            key_topics=l1_data.get("key_topics", []),
            aggregated_preferences=l1_data.get("aggregated_preferences", {}),
            key_entities=l1_data.get("key_entities", []),
            language=l1_data.get("language", "auto"),
            compression_mode=mode,
            compression_model=self._config.compression_model_override or "",
            compression_provider=self._config.compression_provider_override or "",
            compression_latency_ms=latency_ms,
        )

        # Build Layer 2 (if enabled)
        layer_2: Optional[Layer2] = None
        if profile.include_layer2 and l2_data:
            l2_version = (state.layer_2.version + 1) if state.layer_2 else 1
            layer_2 = Layer2(
                session_id=session_id,
                version=l2_version,
                session_summary=l2_data.get("session_summary", ""),
                persistent_user_traits=l2_data.get("persistent_user_traits", {}),
                topic_arc=l2_data.get("topic_arc", []),
                interaction_patterns=l2_data.get("interaction_patterns", {}),
                accumulated_knowledge=l2_data.get("accumulated_knowledge", []),
                unresolved_threads=l2_data.get("unresolved_threads", []),
                source_layer1_versions=[new_version],
                total_turns_compressed=len(batch) + (
                    state.layer_2.total_turns_compressed if state.layer_2 else 0
                ),
                compression_model=self._config.compression_model_override or "",
                compression_provider=self._config.compression_provider_override or "",
            )

        # Update session state
        state.layer_1 = layer_1
        if layer_2:
            state.layer_2 = layer_2
        state.total_compressions += 1
        state.last_compression_turn = batch[-1].turn_number

        logger.info(
            "[CCE] Compression complete | session=%s L1v=%d L2v=%s consumed=%d latency=%.1fms",
            session_id,
            layer_1.version,
            layer_2.version if layer_2 else "N/A",
            len(batch),
            latency_ms,
        )

        return CompressionResult(
            success=True,
            layer_1=layer_1,
            layer_2=layer_2,
            sub_layers_consumed=len(batch),
            compression_mode=mode,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def get_compressed_context(
        self,
        session_id: str,
        *,
        include_layer0_recent: int = 3,
        include_layer1: bool = True,
        include_layer2: bool = True,
    ) -> Dict[str, Any]:
        """
        Assemble a compressed context payload for LLM consumption.

        Returns a dict with available layers formatted for injection
        into a system/user prompt.

        Args:
            session_id: Target session
            include_layer0_recent: Number of recent sub-layers to include
            include_layer1: Whether to include Layer 1
            include_layer2: Whether to include Layer 2

        Returns:
            Dict with layer data and metadata
        """
        state = self.get_session(session_id)
        if state is None:
            logger.debug("[CCE] get_compressed_context: session %s not found", session_id)
            return {"session_id": session_id, "available": False}

        result: Dict[str, Any] = {
            "session_id": session_id,
            "available": True,
            "total_sub_layers": state.layer_0.count,
            "total_compressions": state.total_compressions,
        }

        # Recent sub-layers (Layer 0)
        if include_layer0_recent and state.layer_0.sub_layers:
            recent = state.layer_0.sub_layers[-include_layer0_recent:]
            result["layer_0_recent"] = [
                sl.model_dump(exclude={"sub_layer_id"}) for sl in recent
            ]

        # Layer 1
        if include_layer1 and state.layer_1:
            result["layer_1"] = state.layer_1.model_dump(
                exclude={"layer_id", "compression_model", "compression_provider"},
            )

        # Layer 2
        if include_layer2 and state.layer_2:
            result["layer_2"] = state.layer_2.model_dump(
                exclude={"layer_id", "compression_model", "compression_provider"},
            )

        logger.debug(
            "[CCE] get_compressed_context | session=%s layers=%s",
            session_id,
            ", ".join(k for k in ("layer_0_recent", "layer_1", "layer_2") if k in result),
        )
        return result

    # ------------------------------------------------------------------
    # Full lifecycle helper
    # ------------------------------------------------------------------

    async def process_turn(
        self,
        session_id: str,
        turn_number: int,
        query: str,
        response: str,
        profile_type: ProfileType = ProfileType.CHAT,
        *,
        auto_compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a complete interaction turn:
          1. Create sub-layer
          2. Check if compression threshold reached
          3. Compress if needed
          4. Return updated context

        This is the primary entry-point for integrators.

        Args:
            session_id: Session identifier
            turn_number: Current turn (0-based)
            query: User query
            response: Assistant response
            profile_type: Compression profile
            auto_compress: Whether to auto-trigger compression

        Returns:
            Dict with sub_layer, compression_result (if any), and context
        """
        logger.info(
            "[CCE] process_turn START | session=%s turn=%d profile=%s",
            session_id, turn_number, profile_type.value,
        )

        sub_layer = await self.create_sub_layer(
            session_id=session_id,
            turn_number=turn_number,
            query=query,
            response=response,
            profile_type=profile_type,
        )

        compression_result = None
        if auto_compress and self.should_compress(session_id, profile_type):
            compression_result = await self.compress(
                session_id=session_id,
                profile_type=profile_type,
            )

        context = self.get_compressed_context(session_id)

        logger.info(
            "[CCE] process_turn END | session=%s turn=%d compressed=%s",
            session_id, turn_number,
            compression_result.success if compression_result else "N/A",
        )

        return {
            "sub_layer": sub_layer.model_dump(),
            "compression_result": (
                compression_result.model_dump() if compression_result else None
            ),
            "context": context,
        }

    # ------------------------------------------------------------------
    # State export/import (for persistence integration)
    # ------------------------------------------------------------------

    def export_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export full compression state as serialisable dict."""
        state = self.get_session(session_id)
        if state is None:
            return None
        return state.model_dump()

    def import_state(self, data: Dict[str, Any]) -> str:
        """Import compression state from dict. Returns session_id."""
        state = CompressionState(**data)
        self._sessions[state.session_id] = state
        logger.info("[CCE] State imported for session=%s", state.session_id)
        return state.session_id

    # ------------------------------------------------------------------
    # Configuration access
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as dict."""
        return self._config.model_dump()

    def get_profile_settings(self, profile_type: ProfileType) -> Dict[str, Any]:
        """Return settings for a specific profile."""
        return self._config.get_profile(profile_type).model_dump()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM output, stripping markdown fences if present.
        """
        text = raw.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return json.loads(text)
