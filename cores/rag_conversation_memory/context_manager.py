"""
Structured Memory v4.2.0 - Context Manager

Thread-Based Summary with Smart Promote & Gentle Decay.

Responsibilities:
- Extract focus/key_facts/importance from new messages via LLM
- Deterministic fading with importance-weighted decay
- Smart merge (adjacent + non-adjacent within window)
- Smart promote (reactivate old topics)
- Dual current (CURRENT + HOLD) for topic ping-pong
- Explicit reset detection (IT/EN patterns)
- Archival of faded turns

v4.2.0: Thread-based structured summary
v2.0.0: Initial implementation for FEAT-MEM-002
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import json
import logging
import os
import re

from .models import (
    MemoryState, StructuredContext, Topic, ContextResult, ConversationTurn
)

logger = logging.getLogger(__name__)


# =============================================================================
# DETAIL LEVEL THRESHOLDS
# =============================================================================

DETAIL_LEVELS = [
    # (max_turns_absent_effective, level, max_chars)
    (0, "full", 500),
    (1, "high", 300),
    (2, "recent", 200),
    (5, "fading", 80),
    (10, "background", 30),
]

# Keywords that boost importance floor
CRITICAL_KEYWORDS = {
    "auth", "security", "error", "crash", "bug", "fail",
    "password", "token", "permission", "vulnerability",
    "autenticazione", "sicurezza", "errore", "crash", "bug",
}

# Explicit reset patterns
RESET_PATTERNS_IT = re.compile(
    # MEM-011 v17.15: avoid false positives on "dimentica" used inside
    # "ho dimenticato la password", "non dimenticare di...", etc. The
    # negative lookahead requires "dimentica" to be followed by clause
    # boundaries (end / punctuation / connectives), not an object.
    r"(?i)\b(nuovo argomento|cambiamo topic|"
    r"dimentica(?!\s+(la|il|lo|i|le|un|una|che|come|cosa|quando|dove|"
    r"perch[eé]|quale|mio|mia|miei|mie|tuo|tua|tuoi|tue|password|"
    r"pin|email|codice|nome|numero))|"
    r"ripartiamo|lasciamo perdere|non mi interessa pi[uù]|"
    r"parliamo d'altro|cambiamo discorso)\b"
)
RESET_PATTERNS_EN = re.compile(
    r"(?i)\b(new topic|"
    r"forget(?!\s+(my|your|his|her|their|its|the|a|an|that|this|those|"
    r"these|about|how|what|when|where|why|password|pin|email|"
    r"name|number|code))|"
    r"start over|let'?s move on|"
    r"different subject|change topic|never ?mind|moving on)\b"
)


# =============================================================================
# COMPRESSION PROMPT (v4.2.0 - Thread-based)
# =============================================================================

COMPRESSION_PROMPT = """You are a conversation memory analyst. Extract the core topic and facts from the new messages.

CRITICAL LANGUAGE RULE: Extract focus, key_facts, and anchor_sentence in the SAME LANGUAGE as the user's messages. If the user writes in Italian, all extracted content MUST be in Italian. If in English, extract in English. Detect the user's language from the NEW MESSAGES section below.

CURRENT MEMORY STATE:
{current_state_json}

NEW MESSAGES (only analyze these):
{new_messages}

INSTRUCTIONS:
1. Identify the SUB-TOPIC (focus) of the new messages. Be specific: not "system architecture" but "Redis override system" or "Docker container startup".
2. Extract KEY FACTS as a dense comma-separated string (max 150 chars). Include: names, numbers, technical terms, relationships, decisions.
3. Rate IMPORTANCE 0-10 (how critical is this information for future turns).
4. Check if the new focus MATCHES an existing topic in the thread (exact or very similar). If so, report matched_existing_topic.
5. If resuming an old topic, write a short anchor_sentence bridging old context to new.
6. Extract/update ENTITIES and detect user INTENT.

OUTPUT: Return ONLY valid JSON (no explanations):
{{
  "focus": "specific sub-topic label",
  "key_facts": "dense comma-separated facts from new messages only",
  "importance": 5,
  "matched_existing_topic": null,
  "anchor_sentence": "",
  "entities": {{}},
  "intent": "question|request|information|exploration|progressive_deepdive|greeting|other",
  "confidence": 0.8
}}
"""

# Legacy prompt kept for backward compat (quick_topic_check)
TOPIC_DETECTION_PROMPT = """Analyze if the user changed topic between these messages.

PREVIOUS CONTEXT:
Topic: {previous_topic}
Summary: {previous_summary}

NEW MESSAGE:
{new_message}

Answer with JSON only:
{{
  "topic_changed": true|false,
  "new_topic": "topic name if changed, otherwise empty",
  "confidence": 0.0-1.0
}}
"""


class ContextManager:
    """
    Manages conversation context with thread-based compression and topic tracking.

    v4.2.0: Thread-based structured summary with:
    - Importance-weighted asymmetric fading
    - Smart merge (adjacent + non-adjacent within window)
    - Smart promote (reactivate archived/faded topics)
    - Dual current (CURRENT + HOLD) for ping-pong
    - Explicit reset detection
    - Deterministic lifecycle (LLM does content, code does lifecycle)
    """

    def __init__(
        self,
        di_container: Optional[Any] = None,
        provider_name: Optional[str] = None,
        settings: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
    ):
        self._di_container = di_container
        self._provider_name = provider_name
        self.settings = settings

        self._llm_provider = llm_provider
        self._llm_resolved = llm_provider is not None

        # v6.0.1: Model resolution removed — inference module resolves from ProviderInventory
        # v4.3.0: Resolved by ProviderMapper.resolve_llm_with_fallback()
        self._resolved_provider: Optional[str] = None

        # Settings with defaults
        self._decay_turns = settings.topic_decay_turns if settings else 5
        self._max_previous_topics = settings.max_previous_topics if settings else 3
        self._max_context_tokens = settings.max_context_tokens if settings else 4000
        self._compression_threshold = settings.compression_threshold if settings else 0.8
        self._summary_max_tokens = settings.summary_max_tokens if settings else 1500

        # v4.2.0 thread settings
        self._max_thread_turns = getattr(settings, 'max_thread_turns', 20) if settings else 20
        self._max_archived_turns = getattr(settings, 'max_archived_turns', 30) if settings else 30
        self._fading_chars = getattr(settings, 'fading_key_facts_chars', 80) if settings else 80
        self._background_chars = getattr(settings, 'background_key_facts_chars', 30) if settings else 30
        self._boost_turns = getattr(settings, 'reactivation_boost_turns', 5) if settings else 5
        self._hold_max_turns = getattr(settings, 'hold_max_turns', 3) if settings else 3
        self._soft_merge_window = getattr(settings, 'soft_merge_window', 4) if settings else 4

        # v6.8.x: Compression LLM max_tokens — env override for tuning
        _env_mt = os.environ.get("UBP_MEMORY__COMPRESS_MAX_TOKENS", "")
        self._compress_max_tokens = int(_env_mt) if _env_mt.isdigit() else 1500

    def _detect_language(self, text: str) -> str:
        """Detect language using marker words (same pattern as pipeline_router)."""
        sample = text[:1000].lower()
        indicators = {
            "it": ['che', 'per', 'con', 'della', 'sono', 'questo', 'essere', 'come', 'cosa'],
            "en": ['the', 'and', 'for', 'that', 'this', 'with', 'from', 'what', 'how'],
        }
        scores = {lang: sum(1 for w in words if f' {w} ' in f' {sample} ')
                  for lang, words in indicators.items()}
        return max(scores, key=scores.get)

    # =========================================================================
    # LLM RESOLUTION (lazy)
    # =========================================================================

    async def _get_llm(self) -> Optional[Any]:
        """
        Lazy-resolve LLM module via ProviderMapper centralized fallback chain.

        v4.3.0: Delegates to ProviderMapper.resolve_llm_with_fallback() which
        handles provider -> module resolution, model name auto-prefix, and
        multi-level fallback (Level 1: primary, Level 2a: rag fallback,
        Level 2b: chat fallback). If all levels fail, returns None and the
        caller activates deterministic fallback (Level 3, no LLM).
        """
        if self._llm_resolved:
            return self._llm_provider

        if not self._di_container or not self._provider_name:
            logger.warning("[MEMORY] No DI container or provider name configured")
            self._llm_resolved = True
            return None

        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper

            result = await ProviderMapper.resolve_llm_with_fallback(
                provider_name=self._provider_name,
                di_container=self._di_container,
                model_string=None,  # v6.0.1: model resolved by inference module
            )

            if result:
                self._llm_provider = result["llm_module"]
                self._resolved_provider = result["provider_name"]
                self._llm_resolved = True

                if result["fallback_used"]:
                    logger.warning(
                        f"[MEMORY] LLM resolved via FALLBACK Level {result['fallback_level']}: "
                        f"requested='{self._provider_name}', "
                        f"used='{result['provider_name']}', "
                        f"module='{result['module_name']}', "
                        f"model='{result['model_string'] or '(default)'}'"
                    )
                else:
                    logger.info(
                        f"[MEMORY] LLM resolved: "
                        f"provider='{result['provider_name']}', "
                        f"module='{result['module_name']}', "
                        f"model='{result['model_string'] or '(default)'}'"
                    )
                return self._llm_provider
            else:
                logger.warning(
                    "[MEMORY] No LLM provider available (all fallback levels exhausted). "
                    "Compression will use deterministic fallback (Level 3)."
                )
                self._llm_resolved = True
                return None

        except Exception as e:
            logger.warning(f"[MEMORY] LLM resolution failed unexpectedly: {e}")
            self._llm_resolved = True
            return None

    @property
    def llm_provider(self) -> Optional[Any]:
        return self._llm_provider

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    async def check_and_compress(
        self,
        current_state: MemoryState,
        new_messages: List[Dict[str, Any]],
        force: bool = False
    ) -> Tuple[MemoryState, bool]:
        """
        Check if compression is needed and perform it.

        v4.2.0: Always performs thread-based compression when force=True.
        """
        if not new_messages:
            return current_state, False

        estimated_tokens = self._estimate_tokens(current_state, new_messages)
        threshold_tokens = self._max_context_tokens * self._compression_threshold
        threshold_exceeded = estimated_tokens >= threshold_tokens
        needs_compression = force or threshold_exceeded

        if not needs_compression:
            current_state.turn_counter += len(new_messages)
            removed_topics = current_state.apply_decay(self._decay_turns)
            if removed_topics:
                logger.debug(f"Removed {len(removed_topics)} stale topics")
            return current_state, False

        if force and not threshold_exceeded:
            logger.info(
                f"Compression triggered (force=True): "
                f"{estimated_tokens} tokens (threshold={threshold_tokens}, not exceeded)"
            )
        else:
            logger.info(
                f"Compression triggered: {estimated_tokens} tokens >= {threshold_tokens} threshold"
            )

        try:
            new_state, topic_shifted = await self._compress_with_llm(
                current_state, new_messages
            )
            return new_state, topic_shifted
        except Exception as e:
            logger.error(f"LLM compression failed: {e}, using fallback")
            return self._fallback_compression(current_state, new_messages), False

    # =========================================================================
    # THREAD-BASED COMPRESSION
    # =========================================================================

    async def _compress_with_llm(
        self,
        current_state: MemoryState,
        new_messages: List[Dict[str, Any]]
    ) -> Tuple[MemoryState, bool]:
        """
        Use LLM to extract focus/facts/importance, then apply deterministic lifecycle.
        """
        llm = await self._get_llm()
        if not llm:
            logger.warning("No LLM provider available, using fallback")
            return self._fallback_compression(current_state, new_messages), False

        messages_text = self._format_messages_for_prompt(new_messages)

        # Extract first user query from new messages
        user_query = ""
        for msg in new_messages:
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # Build thread summary for prompt context
        thread_summary = []
        for turn in current_state.conversation_thread:
            thread_summary.append({
                "focus": turn.focus,
                "key_facts": turn.key_facts[:100],
                "detail_level": turn.detail_level,
                "turn_number": turn.turn_number,
            })

        state_for_prompt = {
            "conversation_thread": thread_summary,
            "current_focus": current_state.current_focus,
            "turn_counter": current_state.turn_counter,
            "narrative_summary": current_state.narrative_summary[:200] if current_state.narrative_summary else "",
        }

        prompt = COMPRESSION_PROMPT.format(
            current_state_json=json.dumps(state_for_prompt, indent=2, default=str),
            new_messages=messages_text,
        )

        # MEM-008 v17.15: prompt size guard. If the assembled prompt is
        # absurdly large (e.g., tool result echoed into messages_text or
        # a runaway thread summary), truncate the messages_text section
        # to a hard cap before calling the LLM. Without this guard the
        # compress call will trip the upstream provider's max-context
        # limit, raise, and the caller's fallback compression kicks in
        # AFTER consuming an LLM round-trip and N seconds of latency.
        _MAX_COMPRESS_PROMPT_CHARS = 32000
        if len(prompt) > _MAX_COMPRESS_PROMPT_CHARS:
            logger.warning(
                "[MEM-008] Compression prompt oversize: %d > %d chars, "
                "truncating messages_text section (state preserved)",
                len(prompt), _MAX_COMPRESS_PROMPT_CHARS,
            )
            # Reserve ~80% of the cap for messages_text after stripping
            # state_for_prompt. Recompute prompt with truncated messages.
            overhead = len(prompt) - len(messages_text)
            available_for_messages = max(
                2048, _MAX_COMPRESS_PROMPT_CHARS - overhead - 256
            )
            truncated_msgs = (
                messages_text[:available_for_messages]
                + "\n[...truncated by MEM-008 size guard...]"
            )
            prompt = COMPRESSION_PROMPT.format(
                current_state_json=json.dumps(state_for_prompt, indent=2, default=str),
                new_messages=truncated_msgs,
            )

        # Detect user language and add explicit instruction
        user_texts = " ".join(
            m["content"] for m in new_messages if m.get("role") == "user" and m.get("content")
        )
        detected_lang = self._detect_language(user_texts) if user_texts else "en"
        lang_label = {"it": "Italian", "en": "English"}.get(detected_lang, detected_lang)
        prompt += f"\nDETECTED USER LANGUAGE: {lang_label}. Extract ALL content in {lang_label}.\n"

        try:
            response = await self._call_llm(prompt)
            new_state = self._parse_llm_response(
                response, current_state, new_messages, user_query
            )
            topic_shifted = (
                new_state.current_focus != current_state.current_focus
                and current_state.current_focus is not None
            )
            return new_state, topic_shifted

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def _call_llm(self, prompt: str, max_tokens: int = 0) -> str:
        """
        Call the LLM provider with the resolved provider name.

        v6.0.1: Passes provider= instead of model=. The inference module
        resolves the correct model from ProviderInventory internally.
        v6.8.x: max_tokens configurable via UBP_MEMORY__COMPRESS_MAX_TOKENS
        env var (default 1500). BUG-COMPRESS-001 fix.
        """
        if max_tokens <= 0:
            max_tokens = self._compress_max_tokens
        llm = self._llm_provider
        if not llm:
            raise ValueError("No LLM provider configured")

        extra_kwargs = {}
        if self._resolved_provider:
            extra_kwargs["provider"] = self._resolved_provider

        if hasattr(llm, 'generate'):
            result = await llm.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=0.3, **extra_kwargs
            )
            return result.get('text') or result.get('content') or str(result)
        elif hasattr(llm, 'chat'):
            messages = [{"role": "user", "content": prompt}]
            result = await llm.chat(messages=messages, max_tokens=max_tokens, **extra_kwargs)
            return result.get('content') or result.get('text') or str(result)
        elif hasattr(llm, 'execute'):
            result = await llm.execute(
                operation="generate", prompt=prompt, max_tokens=max_tokens, **extra_kwargs
            )
            return result.get('text') or result.get('content') or str(result)
        else:
            raise ValueError(f"Unknown LLM provider interface: {type(llm)}")

    def _parse_llm_response(
        self,
        response: str,
        current_state: MemoryState,
        new_messages: List[Dict[str, Any]],
        user_query: str = ""
    ) -> MemoryState:
        """
        Parse LLM response and apply deterministic thread lifecycle.
        """
        try:
            data = self._extract_json(response)

            new_focus = data.get('focus', 'general')
            new_key_facts = data.get('key_facts', '')
            importance = data.get('importance', 5)
            matched_existing = data.get('matched_existing_topic', None)
            anchor_sentence = data.get('anchor_sentence', '')
            entities = data.get('entities', {})
            intent = data.get('intent', 'general')
            confidence = data.get('confidence', 0.8)
            suggested_lane = data.get('suggested_lane')
            previous_lane = data.get('previous_lane')
            lane_reason = data.get('lane_reason', '')

            # Importance hardening: floor at 4
            importance = max(importance, 4)

            # Critical keyword boost
            focus_lower = new_focus.lower()
            query_lower = user_query.lower()
            for kw in CRITICAL_KEYWORDS:
                if kw in focus_lower or kw in query_lower:
                    importance = max(importance, 7)
                    break

            return self._apply_thread_update(
                current_state=current_state,
                new_messages=new_messages,
                new_focus=new_focus,
                new_key_facts=new_key_facts,
                importance=importance,
                matched_existing=matched_existing,
                anchor_sentence=anchor_sentence,
                entities=entities,
                intent=intent,
                confidence=confidence,
                user_query=user_query,
                suggested_lane=suggested_lane,
                previous_lane=previous_lane,
                lane_reason=lane_reason,
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            return self._fallback_compression(current_state, new_messages)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response, handling markdown blocks and extra text."""
        text = text.strip()
        # Strip <think>...</think> reasoning blocks (Qwen3)
        text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Markdown json code block
        md_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if md_match:
            try:
                return json.loads(md_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # Generic markdown code block
        md_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
        if md_match:
            try:
                return json.loads(md_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # Outermost JSON object (non-greedy inner matching)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Last resort: greedy match
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # BUG-COMPRESS-001: Try to repair truncated JSON (LLM output cut mid-value)
        # Close open strings and braces so partial responses still yield usable data
        if text.lstrip().startswith('{'):
            repaired = text.rstrip()
            # Close any open string literal (odd quote count)
            quote_count = repaired.count('"') - repaired.count('\\"')
            if quote_count % 2 != 0:
                repaired += '"'
            # Close open braces
            open_braces = repaired.count('{') - repaired.count('}')
            repaired += '}' * max(open_braces, 0)
            try:
                result = json.loads(repaired)
                logger.warning("[COMPRESS] Repaired truncated JSON (%d chars, %d braces closed)", len(text), open_braces)
                return result
            except json.JSONDecodeError:
                pass
        raise ValueError(f"No JSON found in response: {text[:200]}")

    def _apply_thread_update(
        self,
        current_state: MemoryState,
        new_messages: List[Dict[str, Any]],
        new_focus: str,
        new_key_facts: str,
        importance: int,
        matched_existing: Optional[str],
        anchor_sentence: str,
        entities: Dict[str, Any],
        intent: str,
        confidence: float,
        user_query: str,
        suggested_lane: Optional[str] = None,
        previous_lane: Optional[str] = None,
        lane_reason: str = "",
    ) -> MemoryState:
        """
        Apply deterministic thread lifecycle: fading, merge/promote, hold management.
        """
        # Deep copy thread
        thread = [t.model_copy(deep=True) for t in current_state.conversation_thread]
        current_turn = current_state.turn_counter + len(new_messages)

        # Check explicit reset
        is_reset = self._check_explicit_reset(user_query)

        # Step 1: Apply fading to all existing turns
        thread, archived = self._apply_fading(thread, current_turn, current_state.current_focus)

        # Step 2: Merge or promote (skip if explicit reset)
        if is_reset:
            action = "new_topic_reset"
            new_turn = ConversationTurn(
                turn_number=current_turn,
                focus=new_focus,
                key_facts=new_key_facts,
                key_facts_full=new_key_facts,
                detail_level="full",
                importance=importance,
                query=user_query,
                suggested_lane=suggested_lane,
                previous_lane=previous_lane,
                lane_reason=lane_reason,
            )
            thread.append(new_turn)
            logger.info(
                f"[MEMORY-THREAD] action=new_topic_reset focus={new_focus} "
                f"importance={importance} turn={current_turn}"
            )
        else:
            thread, action = self._handle_merge_or_promote(
                thread=thread,
                new_focus=new_focus,
                new_key_facts=new_key_facts,
                importance=importance,
                anchor_sentence=anchor_sentence,
                matched_existing=matched_existing,
                current_turn=current_turn,
                user_query=user_query,
                suggested_lane=suggested_lane,
                previous_lane=previous_lane,
                lane_reason=lane_reason,
            )

        # Step 3: Manage hold (dual current)
        old_focus = current_state.current_focus
        new_current_focus = new_focus
        hold_focus = current_state.hold_focus
        hold_since = current_state.hold_since_turn

        if not is_reset:
            new_current_focus, hold_focus, hold_since = self._manage_hold(
                old_focus=old_focus,
                new_focus=new_focus,
                hold_focus=hold_focus,
                hold_since_turn=hold_since,
                current_turn=current_turn,
            )

        # Step 4: Trim thread to max size
        if len(thread) > self._max_thread_turns:
            excess = thread[:len(thread) - self._max_thread_turns]
            thread = thread[len(thread) - self._max_thread_turns:]
            for t in excess:
                archived.append(t.model_dump(mode='json'))
                logger.info(
                    f"[MEMORY-THREAD] action=trim_to_max focus={t.focus} "
                    f"turn={t.turn_number}"
                )

        # Step 5: Build topic flow
        topic_flow = list(current_state.topic_flow)
        is_resumed = action in ("smart_promote", "soft_merge_non_adjacent")
        topic_flow.append({
            "turn": current_turn,
            "focus": new_focus,
            "resumed": is_resumed,
        })
        # Keep last 30 entries
        if len(topic_flow) > 30:
            topic_flow = topic_flow[-30:]

        topic_progression = self._derive_topic_progression(topic_flow)

        # Step 6: Update structured_context for backward compat
        structured_context = StructuredContext(
            current_topic=new_current_focus or new_focus,
            topic_status="shifting" if action != "merge_adjacent" and old_focus and old_focus != new_focus else "open",
            intent=intent,
            entities=entities,
            confidence=confidence,
        )

        # Step 7: Merge archived
        all_archived = list(current_state.archived_turns) + archived
        if len(all_archived) > self._max_archived_turns:
            all_archived = all_archived[-self._max_archived_turns:]

        # Step 8: Sync legacy previous_topics from thread
        previous_topics = self._sync_previous_topics(thread, new_current_focus)

        # Build new state
        new_state = MemoryState(
            version=current_state.version + 1,
            created_at=current_state.created_at,
            last_updated=datetime.now(timezone.utc),
            turn_counter=current_turn,
            compression_history=current_state.compression_history.copy(),
            # Thread-based
            conversation_thread=thread,
            current_focus=new_current_focus,
            hold_focus=hold_focus,
            hold_since_turn=hold_since,
            topic_flow=topic_flow,
            topic_progression=topic_progression,
            archived_turns=all_archived,
            # Legacy compat
            structured_context=structured_context,
            previous_topics=previous_topics,
        )

        # Derive narrative_summary from thread
        new_state.narrative_summary = new_state.derive_narrative_summary()

        # Estimate tokens
        new_state.token_count = self._estimate_state_tokens(new_state)

        # Log compression event
        new_state.add_compression_event(
            messages_compressed=len(new_messages),
            tokens_saved=current_state.token_count - new_state.token_count if current_state.token_count > 0 else 0,
            trigger="thread_compression"
        )

        return new_state

    # =========================================================================
    # FADING (deterministic, importance-weighted)
    # =========================================================================

    def _apply_fading(
        self,
        thread: List[ConversationTurn],
        current_turn: int,
        current_focus: Optional[str],
    ) -> Tuple[List[ConversationTurn], List[Dict[str, Any]]]:
        """
        Apply importance-weighted fading to all thread entries.

        Returns (updated_thread, archived_turns_dicts).
        """
        archived = []
        remaining = []

        for entry in thread:
            turns_absent = current_turn - entry.turn_number
            if turns_absent <= 0:
                remaining.append(entry)
                continue

            # Importance-weighted effective absence
            turns_absent_effective = turns_absent / (1 + entry.importance / 5.0)

            # Check boost resistance
            if entry.reactivation_boost > 0:
                entry.reactivation_boost -= 1

            # Determine detail level from table
            new_level = "full"
            max_chars = 500
            for threshold, level, chars in DETAIL_LEVELS:
                if turns_absent_effective <= threshold:
                    new_level = level
                    max_chars = chars
                    break
            else:
                # Beyond all thresholds
                if turns_absent_effective > 10:
                    # Check boost resistance for archival
                    if entry.reactivation_boost >= 6:
                        new_level = "background"
                        max_chars = 30
                    else:
                        new_level = "archived"
                        max_chars = 0
                elif turns_absent_effective > 5:
                    if entry.reactivation_boost >= 4:
                        new_level = "fading"
                        max_chars = 80
                    else:
                        new_level = "background"
                        max_chars = 30
                elif turns_absent_effective > 2:
                    if entry.reactivation_boost >= 2:
                        new_level = "recent"
                        max_chars = 200
                    else:
                        new_level = "fading"
                        max_chars = 80

            old_level = entry.detail_level

            if new_level == "archived":
                # Archive this turn
                archived.append(entry.model_dump(mode='json'))
                logger.info(
                    f"[MEMORY-THREAD] action=archive focus={entry.focus} "
                    f"detail_level={old_level}->archived "
                    f"importance={entry.importance} turn={entry.turn_number}"
                )
                continue

            # Apply new level and truncate
            entry.detail_level = new_level
            if max_chars > 0 and len(entry.key_facts) > max_chars:
                entry.key_facts = self._truncate_preserving_words(entry.key_facts, max_chars)

            if old_level != new_level:
                logger.info(
                    f"[MEMORY-THREAD] action=fading focus={entry.focus} "
                    f"detail_level={old_level}->{new_level} "
                    f"boost={entry.reactivation_boost} importance={entry.importance} "
                    f"turn={entry.turn_number} merge_count={entry.merge_count}"
                )

            remaining.append(entry)

        return remaining, archived

    # =========================================================================
    # MERGE / PROMOTE
    # =========================================================================

    def _handle_merge_or_promote(
        self,
        thread: List[ConversationTurn],
        new_focus: str,
        new_key_facts: str,
        importance: int,
        anchor_sentence: str,
        matched_existing: Optional[str],
        current_turn: int,
        user_query: str,
        suggested_lane: Optional[str] = None,
        previous_lane: Optional[str] = None,
        lane_reason: str = "",
    ) -> Tuple[List[ConversationTurn], str]:
        """
        4-step merge/promote logic.
        Returns (updated_thread, action_taken).
        """
        # Step 1: Adjacent merge (same focus, last turn in thread)
        if thread:
            last = thread[-1]
            if (last.focus == new_focus or
                (matched_existing and last.focus == matched_existing)) \
                    and last.turn_number >= current_turn - 2:
                last.focus = new_focus  # Bug fix: sync focus with current_focus after merge
                last.key_facts_full = self._merge_facts(last.key_facts_full, new_key_facts)
                last.key_facts = self._truncate_preserving_words(
                    self._merge_facts(last.key_facts, new_key_facts), 500
                )  # Bug fix: cap key_facts at detail_level "full" max (500 chars)
                last.query = user_query  # Keep latest query for LLM context
                last.turn_number = current_turn
                last.merge_count += 1
                last.importance = max(last.importance, importance)
                last.reactivation_boost = max(last.reactivation_boost, 2)
                last.detail_level = "full"
                last.suggested_lane = suggested_lane
                last.previous_lane = previous_lane
                last.lane_reason = lane_reason
                logger.info(
                    f"[MEMORY-THREAD] action=merge_adjacent focus={new_focus} "
                    f"merge_count={last.merge_count} importance={last.importance} "
                    f"turn={current_turn}"
                )
                return thread, "merge_adjacent"

        # Step 2: Soft merge non-adjacent (same/similar focus, within window)
        similar = self._find_similar_focus(thread, new_focus, matched_existing)
        if similar and (current_turn - similar.turn_number) <= self._soft_merge_window:
            similar.focus = new_focus  # Bug fix: sync focus with current_focus after merge
            similar.key_facts_full = self._merge_facts(similar.key_facts_full, new_key_facts)
            similar.key_facts = self._truncate_preserving_words(
                self._merge_facts(similar.key_facts, new_key_facts), 500
            )  # Bug fix: cap key_facts at detail_level "full" max (500 chars)
            similar.query = user_query  # Keep latest query for LLM context
            similar.turn_number = current_turn
            similar.merge_count += 1
            similar.importance = max(similar.importance, importance)
            similar.reactivation_boost = max(similar.reactivation_boost, 2)
            similar.detail_level = "full"
            similar.suggested_lane = suggested_lane
            similar.previous_lane = previous_lane
            similar.lane_reason = lane_reason
            logger.info(
                f"[MEMORY-THREAD] action=soft_merge_non_adjacent focus={new_focus} "
                f"matched={similar.focus} merge_count={similar.merge_count} "
                f"importance={similar.importance} turn={current_turn}"
            )
            return thread, "soft_merge_non_adjacent"

        # Step 3: Smart promote (match beyond window)
        if similar and (current_turn - similar.turn_number) > self._soft_merge_window:
            old_level = similar.detail_level
            similar.key_facts_full = self._merge_facts(similar.key_facts_full, new_key_facts)
            similar.key_facts = self._truncate_preserving_words(
                similar.key_facts_full,
                self._promote_chars(similar.detail_level)
            )
            similar.detail_level = self._promote_level(similar.detail_level)
            similar.is_resumed = True
            similar.anchor_sentence = anchor_sentence
            similar.reactivation_boost = self._boost_turns
            similar.query = user_query  # Keep latest query for LLM context
            similar.turn_number = current_turn
            similar.importance = max(similar.importance, importance)
            similar.suggested_lane = suggested_lane
            similar.previous_lane = previous_lane
            similar.lane_reason = lane_reason
            logger.info(
                f"[MEMORY-THREAD] action=smart_promote focus={similar.focus} "
                f"detail_level={old_level}->{similar.detail_level} "
                f"boost={similar.reactivation_boost} importance={similar.importance} "
                f"turn={current_turn}"
            )
            return thread, "smart_promote"

        # Step 4: New topic
        new_turn = ConversationTurn(
            turn_number=current_turn,
            focus=new_focus,
            key_facts=new_key_facts,
            key_facts_full=new_key_facts,
            detail_level="full",
            importance=importance,
            query=user_query,
            suggested_lane=suggested_lane,
            previous_lane=previous_lane,
            lane_reason=lane_reason,
        )
        thread.append(new_turn)
        logger.info(
            f"[MEMORY-THREAD] action=new_topic focus={new_focus} "
            f"importance={importance} turn={current_turn}"
        )
        return thread, "new_topic"

    def _find_similar_focus(
        self,
        thread: List[ConversationTurn],
        new_focus: str,
        matched_existing: Optional[str],
    ) -> Optional[ConversationTurn]:
        """Find existing thread entry with same or matched focus."""
        focus_lower = new_focus.lower().strip()

        # First try exact match
        for turn in reversed(thread):
            if turn.focus.lower().strip() == focus_lower:
                return turn

        # Then try LLM-reported match
        if matched_existing:
            matched_lower = matched_existing.lower().strip()
            for turn in reversed(thread):
                if turn.focus.lower().strip() == matched_lower:
                    return turn

        return None

    def _merge_facts(self, existing: str, new_facts: str) -> str:
        """Merge facts using bullet separator."""
        if not existing:
            return new_facts
        if not new_facts:
            return existing
        return f"{existing} | {new_facts}"

    def _promote_level(self, current_level: str) -> str:
        """Promote detail level up one step."""
        promotion = {
            "archived": "background",
            "background": "fading",
            "fading": "recent",
            "recent": "high",
            "high": "full",
            "full": "full",
        }
        return promotion.get(current_level, "full")

    def _promote_chars(self, current_level: str) -> int:
        """Get max chars after promotion."""
        chars_map = {
            "archived": 30,
            "background": 80,
            "fading": 200,
            "recent": 300,
            "high": 500,
            "full": 500,
        }
        promoted = self._promote_level(current_level)
        return chars_map.get(promoted, 500)

    # =========================================================================
    # HOLD MANAGEMENT (dual current)
    # =========================================================================

    def _manage_hold(
        self,
        old_focus: Optional[str],
        new_focus: str,
        hold_focus: Optional[str],
        hold_since_turn: int,
        current_turn: int,
    ) -> Tuple[str, Optional[str], int]:
        """
        Manage CURRENT + HOLD dual pointers.
        Returns (new_current_focus, new_hold_focus, new_hold_since_turn).
        """
        # If returning to hold_focus -> swap
        if hold_focus and new_focus.lower().strip() == hold_focus.lower().strip():
            logger.info(
                f"[MEMORY-THREAD] action=hold_swap "
                f"current={old_focus}->{new_focus} hold={hold_focus}->{old_focus}"
            )
            return new_focus, old_focus, current_turn

        # If focus changed, move old to hold
        if old_focus and old_focus.lower().strip() != new_focus.lower().strip():
            # Clear hold if expired
            if hold_focus and (current_turn - hold_since_turn) > self._hold_max_turns:
                logger.info(
                    f"[MEMORY-THREAD] action=hold_expired hold={hold_focus} "
                    f"after {current_turn - hold_since_turn} turns"
                )
                hold_focus = None

            new_hold = old_focus
            logger.info(
                f"[MEMORY-THREAD] action=hold_set current={new_focus} hold={new_hold}"
            )
            return new_focus, new_hold, current_turn

        # Focus unchanged
        # Expire hold if needed
        if hold_focus and (current_turn - hold_since_turn) > self._hold_max_turns:
            logger.info(
                f"[MEMORY-THREAD] action=hold_expired hold={hold_focus}"
            )
            return new_focus, None, 0

        return new_focus, hold_focus, hold_since_turn

    # =========================================================================
    # EXPLICIT RESET DETECTION
    # =========================================================================

    @staticmethod
    def _check_explicit_reset(query: str) -> bool:
        """Check if query contains explicit topic reset patterns."""
        if not query:
            return False
        if RESET_PATTERNS_IT.search(query):
            return True
        if RESET_PATTERNS_EN.search(query):
            return True
        return False

    # =========================================================================
    # TOPIC PROGRESSION
    # =========================================================================

    @staticmethod
    def _derive_topic_progression(topic_flow: List[Dict[str, Any]]) -> str:
        """Derive flat string from topic flow list."""
        if not topic_flow:
            return ""

        parts = []
        prev_focus = None
        for entry in topic_flow:
            focus = entry.get("focus", "")
            resumed = entry.get("resumed", False)
            if focus == prev_focus:
                continue
            if resumed:
                parts.append(f"{focus} (resumed)")
            else:
                parts.append(focus)
            prev_focus = focus

        return " -> ".join(parts) if parts else ""

    # =========================================================================
    # TEXT TRUNCATION
    # =========================================================================

    @staticmethod
    def _truncate_preserving_words(text: str, max_chars: int) -> str:
        """Truncate text at word boundary."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.5:
            truncated = truncated[:last_space]
        return truncated.rstrip(' ,|')

    # =========================================================================
    # SYNC PREVIOUS TOPICS (backward compat)
    # =========================================================================

    def _sync_previous_topics(
        self,
        thread: List[ConversationTurn],
        current_focus: Optional[str],
    ) -> List[Topic]:
        """Derive legacy previous_topics from thread for backward compat."""
        topics = []
        for turn in reversed(thread):
            if current_focus and turn.focus == current_focus:
                continue
            if turn.detail_level in ("background", "fading"):
                status = "abandoned"
            else:
                status = "open"
            topics.append(Topic(
                topic=turn.focus,
                status=status,
                key_info=turn.key_facts[:100],
                decay_remaining=max(1, 5 - (len(thread) - thread.index(turn))),
                last_mentioned_turn=turn.turn_number,
            ))
            if len(topics) >= self._max_previous_topics:
                break
        return topics

    # =========================================================================
    # FALLBACK COMPRESSION
    # =========================================================================

    def _fallback_compression(
        self,
        current_state: MemoryState,
        new_messages: List[Dict[str, Any]]
    ) -> MemoryState:
        """Simple fallback when LLM unavailable."""
        user_messages = [m['content'] for m in new_messages if m.get('role') == 'user']
        new_content = " ".join(user_messages)

        # Try to create a thread entry from user messages
        thread = [t.model_copy(deep=True) for t in current_state.conversation_thread]
        current_turn = current_state.turn_counter + len(new_messages)

        if new_content.strip():
            # Use first 50 chars as focus
            focus = new_content[:50].strip()
            if len(new_content) > 50:
                focus = focus.rsplit(' ', 1)[0] + "..."

            new_turn = ConversationTurn(
                turn_number=current_turn,
                focus=focus,
                key_facts=new_content[:200],
                key_facts_full=new_content,
                detail_level="full",
                importance=5,
                query=new_content[:500],
            )
            thread.append(new_turn)

        # Build narrative summary
        if current_state.narrative_summary:
            narrative = f"{current_state.narrative_summary} {new_content}"
        else:
            narrative = new_content

        max_chars = self._summary_max_tokens * 4
        if len(narrative) > max_chars:
            narrative = narrative[-max_chars:]

        new_state = MemoryState(
            version=current_state.version + 1,
            created_at=current_state.created_at,
            last_updated=datetime.now(timezone.utc),
            turn_counter=current_turn,
            narrative_summary=narrative,
            structured_context=current_state.structured_context,
            previous_topics=current_state.previous_topics.copy(),
            compression_history=current_state.compression_history.copy(),
            conversation_thread=thread,
            current_focus=current_state.current_focus or (thread[-1].focus if thread else None),
            hold_focus=current_state.hold_focus,
            hold_since_turn=current_state.hold_since_turn,
            topic_flow=current_state.topic_flow.copy(),
            archived_turns=current_state.archived_turns.copy(),
        )

        new_state.apply_decay(self._decay_turns)
        new_state.token_count = self._estimate_state_tokens(new_state)
        new_state.add_compression_event(
            messages_compressed=len(new_messages),
            tokens_saved=0,
            trigger="fallback"
        )

        return new_state

    # =========================================================================
    # TOKEN ESTIMATION
    # =========================================================================

    def _estimate_tokens(
        self,
        state: MemoryState,
        new_messages: List[Dict[str, Any]]
    ) -> int:
        state_tokens = self._estimate_state_tokens(state)
        message_chars = sum(len(m.get('content', '')) for m in new_messages)
        return state_tokens + message_chars // 4

    def _estimate_state_tokens(self, state: MemoryState) -> int:
        total_chars = len(state.narrative_summary)
        total_chars += len(state.structured_context.current_topic)
        total_chars += len(str(state.structured_context.entities))

        for topic in state.previous_topics:
            total_chars += len(topic.topic) + len(topic.key_info)

        for turn in state.conversation_thread:
            total_chars += len(turn.focus) + len(turn.key_facts)

        return total_chars // 4

    # =========================================================================
    # MESSAGE FORMATTING
    # =========================================================================

    def _format_messages_for_prompt(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    # =========================================================================
    # QUICK TOPIC CHECK (legacy, kept for compat)
    # =========================================================================

    async def quick_topic_check(
        self,
        current_topic: str,
        current_summary: str,
        new_message: str
    ) -> Tuple[bool, str, float]:
        if not self._llm_provider:
            if current_topic.lower() in new_message.lower():
                return False, "", 0.9
            return True, "", 0.3

        prompt = TOPIC_DETECTION_PROMPT.format(
            previous_topic=current_topic,
            previous_summary=current_summary[:500],
            new_message=new_message
        )

        try:
            response = await self._call_llm(prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return (
                    data.get('topic_changed', False),
                    data.get('new_topic', ''),
                    data.get('confidence', 0.5)
                )
        except Exception as e:
            logger.warning(f"Quick topic check failed: {e}")

        return False, "", 0.5

    # =========================================================================
    # BUILD CONTEXT RESULT
    # =========================================================================

    def build_context_result(
        self,
        state: MemoryState,
        raw_messages: List[Dict[str, Any]]
    ) -> ContextResult:
        """Build ContextResult from state and raw messages."""
        return ContextResult(
            raw_messages=raw_messages,
            narrative_summary=state.derive_narrative_summary(),
            structured_context=state.structured_context,
            previous_topics=state.previous_topics,
            has_structured_context=True,
            topic_shifting=state.structured_context.is_shifting(),
            # v4.2.0 thread fields
            conversation_thread=state.conversation_thread,
            current_focus=state.current_focus,
            hold_focus=state.hold_focus,
            topic_progression=state.topic_progression,
        )
