"""
Framework-agnostic HIAMS provider.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from .episodic_manager import AdaptiveEpisodicManager
from .models import (
    CompressionResult,
    HIAMSConfig,
    HIAMSSessionState,
    ProcessingResult,
    ProfileType,
)
from .projection import QueryAwareProjectionBuilder
from .slot_manager import ProtectedSlotManager

logger = logging.getLogger(__name__)


class HIAMSProvider:
    """Main provider for HIAMS."""

    def __init__(
        self,
        config: Dict[str, Any],
        di_container: Any = None,
        llm_adapter: Any = None,
    ):
        self._raw_config = config
        self._config = HIAMSConfig(**config)
        self._di_container = di_container
        self._llm_provider = llm_adapter
        self._sessions: Dict[str, HIAMSSessionState] = {}
        self._slots = ProtectedSlotManager()
        self._episodic = AdaptiveEpisodicManager()
        self._projection = QueryAwareProjectionBuilder()

        logger.info(
            "[HIAMS] Provider initialised | chat_trigger=%d | agent_trigger=%d | enabled=%s",
            self._config.chat_profile.compression_trigger_threshold,
            self._config.agent_loop_profile.compression_trigger_threshold,
            self._config.enabled,
        )

    def get_or_create_session(self, session_id: str) -> HIAMSSessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = HIAMSSessionState(session_id=session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[HIAMSSessionState]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        removed = self._sessions.pop(session_id, None)
        return removed is not None

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

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
        session = self.get_or_create_session(session_id)
        profile = self._config.get_profile(profile_type)

        slot_update = self._slots.update_slots(session.structured_slots, query, response)
        session.structured_slots = slot_update.slots

        snapshot = self._episodic.create_snapshot(turn_number, query, response, slot_update)
        self._episodic.add_snapshot(session, snapshot, profile)

        compression_result = None
        if auto_compress:
            needed, reason = self._episodic.should_compress(session, profile)
            if needed:
                compression_result = await self.compress(
                    session_id=session_id,
                    profile_type=profile_type,
                    force=False,
                    reason=reason,
                )

        projection = self._projection.build(session, query, profile)

        session.total_turns = max(session.total_turns, turn_number + 1)
        session.query_history.append(query)
        session.projection_history.append(projection.intent)

        result = ProcessingResult(
            structured_slots=session.structured_slots,
            slot_update=slot_update,
            snapshot=snapshot,
            compression_result=compression_result,
            projected_context=projection,
        )
        return result.model_dump()

    def should_compress(
        self,
        session_id: str,
        profile_type: ProfileType = ProfileType.CHAT,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            return {"session_id": session_id, "needs_compression": False, "reason": "session-not-found"}

        needed, reason = self._episodic.should_compress(session, self._config.get_profile(profile_type))
        return {"session_id": session_id, "needs_compression": needed, "reason": reason}

    async def compress(
        self,
        session_id: str,
        profile_type: ProfileType = ProfileType.CHAT,
        *,
        force: bool = False,
        reason: str = "",
    ) -> CompressionResult:
        session = self.get_or_create_session(session_id)
        profile = self._config.get_profile(profile_type)

        if not force:
            needed, auto_reason = self._episodic.should_compress(session, profile)
            if not needed:
                return CompressionResult(success=True, compression_triggered=False, reason=auto_reason)
            if not reason:
                reason = auto_reason

        batch = self._episodic.select_batch(session, profile)
        llm_candidate = await self._call_compression_llm(session, batch)
        return self._episodic.compress_batch(
            session=session,
            slots=session.structured_slots,
            batch=batch,
            llm_candidate=llm_candidate,
            profile=profile,
            reason=reason or "manual",
        )

    def get_projected_context(
        self,
        session_id: str,
        query: str,
        profile_type: ProfileType = ProfileType.CHAT,
    ) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            return {"session_id": session_id, "available": False}
        projection = self._projection.build(session, query, self._config.get_profile(profile_type))
        return projection.model_dump()

    def export_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self.get_session(session_id)
        if state is None:
            return None
        return state.model_dump()

    async def _call_compression_llm(
        self,
        session: HIAMSSessionState,
        batch: list[Any],
    ) -> Optional[Dict[str, Any]]:
        llm = await self._get_llm()
        if llm is None or not batch:
            return None

        prompt = (
            "Build a conservative JSON candidate for HIAMS layer1/layer2 merge.\n"
            "Return JSON with keys: conversation_focus, topic, active_threads, key_facts, "
            "entity_index, numeric_facts, event_refs, order_refs, directions, "
            "negative_decisions, user_preferences, quality_score.\n\n"
            f"STRUCTURED_SLOTS:\n{json.dumps(session.structured_slots.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"BATCH:\n{json.dumps([snap.model_dump() for snap in batch], ensure_ascii=False, indent=2)}"
        )

        raw = None
        try:
            if hasattr(llm, "generate"):
                result = await llm.generate(prompt=prompt, max_tokens=900, temperature=0.1)
                raw = result.get("text") or result.get("response") or result.get("content")
            elif hasattr(llm, "chat"):
                result = await llm.chat(messages=[{"role": "user", "content": prompt}], max_tokens=900, temperature=0.1)
                raw = result.get("text") or result.get("response") or result.get("content")
        except Exception as exc:
            logger.warning("[HIAMS] Compression LLM call failed: %s", exc)
            return None

        if not raw:
            return None

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return None
            return parsed
        except Exception:
            logger.debug("[HIAMS] Compression LLM output was not valid JSON")
            return None

    async def _get_llm(self) -> Any:
        if self._llm_provider is not None:
            return self._llm_provider
        if self._di_container is None:
            return None

        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper

            explicit = self._config.compression_provider_override or ""
            candidates = []
            if explicit and explicit in ProviderMapper.PROVIDER_MAP:
                candidates.append(ProviderMapper.PROVIDER_MAP[explicit])
            for module_name, provider_name in ProviderMapper.resolve_chain("enrichment"):
                entry = (module_name, provider_name)
                if entry not in candidates:
                    candidates.append(entry)
            for module_name, _provider_name in candidates:
                try:
                    llm_module = await self._di_container.resolve(module_name)
                    if llm_module:
                        self._llm_provider = llm_module
                        return llm_module
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("[HIAMS] LLM resolution failed: %s", exc)
        return None
