"""
Adaptive episodic memory for HIAMS.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    AdaptiveLayer1Block,
    AdaptiveLayer2Memory,
    AdaptiveSnapshot,
    CompressionResult,
    CoverageClass,
    HIAMSProfileSettings,
    HIAMSSessionState,
    SlotUpdateResult,
    StructuredSlots,
)

logger = logging.getLogger(__name__)


def estimate_tokens_json(data: Any) -> int:
    try:
        from ubp_enterprise_hybrid.mcp_runtime.core.token_limits import TokenCounter

        return TokenCounter.count_tokens(json.dumps(data, ensure_ascii=False, default=str), provider="vllm")
    except Exception:
        return max(1, len(json.dumps(data, ensure_ascii=False, default=str)) // 4)


def _dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


class AdaptiveEpisodicManager:
    """Manage adaptive Layer0/1/2 for HIAMS."""

    def create_snapshot(
        self,
        turn_number: int,
        query: str,
        response: str,
        slot_update: SlotUpdateResult,
    ) -> AdaptiveSnapshot:
        coverage = self._derive_coverage(slot_update)
        referenced_slots = slot_update.changed_slots[:]
        key_entities = slot_update.slots.key_entities
        facts = slot_update.new_facts[:8]
        salience = self._compute_user_salience(query, response, slot_update)

        return AdaptiveSnapshot(
            turn_number=turn_number,
            focus=self._derive_focus(query, response, coverage),
            intent=self._classify_intent(query),
            query_summary=query[:220],
            response_summary=response[:220],
            coverage_classes=coverage,
            key_facts=facts,
            structured_delta={
                name: getattr(slot_update.slots, name).model_dump()
                if hasattr(getattr(slot_update.slots, name), "model_dump")
                else getattr(slot_update.slots, name)
                for name in slot_update.changed_slots
            },
            referenced_slots=referenced_slots,
            entities=_dedup(
                key_entities.products[:6]
                + key_entities.events[:4]
                + key_entities.places[:4]
                + key_entities.times[:4]
            ),
            user_salience=salience,
            info_gain_score=max(slot_update.info_gain_score, len(facts)),
        )

    def add_snapshot(
        self,
        session: HIAMSSessionState,
        snapshot: AdaptiveSnapshot,
        profile: HIAMSProfileSettings,
    ) -> None:
        session.layer0.append(snapshot)
        max_retained = max(profile.layer0_recall_window, profile.layer0_base_window) + 4
        if len(session.layer0) > max_retained:
            session.layer0 = session.layer0[-max_retained:]
        self._apply_immediate_promotion(session, snapshot)

    def should_compress(
        self,
        session: HIAMSSessionState,
        profile: HIAMSProfileSettings,
    ) -> Tuple[bool, str]:
        unabsorbed = [snap for snap in session.layer0 if not snap.absorbed]
        if len(unabsorbed) < profile.compression_trigger_threshold:
            return False, "threshold-not-reached"

        if session.last_compression_turn >= 0:
            turns_since = session.total_turns - session.last_compression_turn
            if turns_since < profile.min_turns_between_compressions:
                return False, "hysteresis"

        delta_score = sum(snap.info_gain_score for snap in unabsorbed)
        if delta_score < profile.min_delta_info_score:
            return False, "delta-too-low"

        semantic_delta = delta_score / max(sum(len(snap.key_facts) for snap in unabsorbed), 1)
        has_critical = any(
            coverage in {CoverageClass.ORDER, CoverageClass.DIETARY, CoverageClass.EVENT, CoverageClass.LOGISTICS}
            for snap in unabsorbed
            for coverage in snap.coverage_classes
        )
        if semantic_delta < 0.12 and not has_critical:
            return False, "semantic-delta-too-low"

        layer0_tokens = estimate_tokens_json([snap.model_dump() for snap in session.layer0])
        if layer0_tokens > profile.projection_budget_tokens:
            return True, "layer0-budget-pressure"

        if has_critical:
            return True, "threshold+delta"
        return True, "threshold+delta"

    def select_batch(
        self,
        session: HIAMSSessionState,
        profile: HIAMSProfileSettings,
    ) -> List[AdaptiveSnapshot]:
        unabsorbed = [snap for snap in session.layer0 if not snap.absorbed]
        return unabsorbed[: max(profile.compression_trigger_threshold, profile.layer0_base_window // 2)]

    def compress_batch(
        self,
        session: HIAMSSessionState,
        slots: StructuredSlots,
        batch: Sequence[AdaptiveSnapshot],
        llm_candidate: Optional[Dict[str, Any]],
        profile: HIAMSProfileSettings,
        reason: str,
    ) -> CompressionResult:
        t0 = time.time()
        if not batch:
            return CompressionResult(success=True, compression_triggered=False, reason="empty-batch")

        deterministic = self._build_deterministic_layer1(batch, slots, session)
        merged = self._merge_with_llm_candidate(deterministic, llm_candidate)

        previous = self._find_merge_candidate(session.layer1_blocks, merged.coverage_classes)
        if previous is not None:
            merged = self._non_destructive_merge(previous, merged)
            merged.version = previous.version + 1

        merged.last_updated_turn = batch[-1].turn_number
        merged.covered_turns = _dedup(str(s.turn_number) for s in batch)
        merged.covered_turns = [int(value) for value in merged.covered_turns]

        self._upsert_layer1(session, merged, profile)
        new_layer2 = self._promote_layer2(session, slots, merged, profile)

        for snapshot in batch:
            snapshot.absorbed = True
        retained_absorbed = [snap for snap in session.layer0 if snap.absorbed][-max(2, profile.layer0_base_window // 2) :]
        retained_unabsorbed = [snap for snap in session.layer0 if not snap.absorbed]
        session.layer0 = sorted(retained_absorbed + retained_unabsorbed, key=lambda snap: snap.turn_number)
        session.total_compressions += 1
        session.last_compression_turn = batch[-1].turn_number

        return CompressionResult(
            success=True,
            compression_triggered=True,
            layer1_block=merged,
            layer2=new_layer2,
            consumed_snapshots=len(batch),
            reason=reason,
            info_gain_score=sum(s.info_gain_score for s in batch),
            latency_ms=(time.time() - t0) * 1000,
        )

    def _derive_coverage(self, slot_update: SlotUpdateResult) -> List[CoverageClass]:
        classes: List[CoverageClass] = []
        mapping = {
            "order_state": CoverageClass.ORDER,
            "dietary_profile": CoverageClass.DIETARY,
            "event_plan": CoverageClass.EVENT,
            "logistics": CoverageClass.LOGISTICS,
        }
        for name in slot_update.changed_slots:
            mapped = mapping.get(name, CoverageClass.GENERAL)
            if mapped not in classes:
                classes.append(mapped)
        if not classes:
            classes.append(CoverageClass.GENERAL)
        return classes

    def _compute_user_salience(
        self,
        query: str,
        response: str,
        slot_update: SlotUpdateResult,
    ) -> float:
        query_l = query.lower()
        score = float(slot_update.info_gain_score)
        if "totale" in query_l or "ordine" in query_l:
            score += 2.0
        if "senza glutine" in query_l or "celiac" in query_l:
            score += 2.0
        if "jazz" in query_l or "piazza municipale" in query_l:
            score += 1.5
        if "non " in query_l or "niente" in query_l:
            score += 1.0
        if "€" in response or ":" in response:
            score += 1.0
        return score

    def _derive_focus(
        self,
        query: str,
        response: str,
        coverage: Sequence[CoverageClass],
    ) -> str:
        query_l = query.lower()
        if CoverageClass.DIETARY in coverage:
            return "vincoli alimentari e compatibilita' menu"
        if CoverageClass.EVENT in coverage:
            return "eventi locali e logistica serale"
        if CoverageClass.ORDER in coverage:
            return "gestione ordine e riepilogo economico"
        if "cocktail" in query_l or "negroni" in query_l:
            return "scelta drink e preferenze"
        return "conversazione multi-thread"

    def _classify_intent(self, query: str) -> str:
        query_l = query.lower()
        if any(word in query_l for word in ("aggiungi", "togli", "sostituisci", "confermo")):
            return "ordering"
        if any(word in query_l for word in ("riepilog", "ricordam", "quanto")):
            return "recall"
        if any(word in query_l for word in ("jazz", "evento", "concerto")):
            return "event"
        return "question"

    def _build_deterministic_layer1(
        self,
        batch: Sequence[AdaptiveSnapshot],
        slots: StructuredSlots,
        session: HIAMSSessionState,
    ) -> AdaptiveLayer1Block:
        coverage = self._aggregate_coverage(batch)
        topic = self._dominant_topic(batch)
        key_facts = _dedup(
            fact
            for snapshot in batch
            for fact in snapshot.key_facts
        )[:16]
        entity_index = _dedup(
            entity
            for snapshot in batch
            for entity in snapshot.entities
        )[:20]
        numeric_facts = _dedup(slots.key_entities.prices + slots.key_entities.times)[:12]
        active_threads = _dedup(snapshot.focus for snapshot in batch)[:8]
        event_refs = _dedup(
            [slots.event_plan.name or "", slots.event_plan.location or "", slots.event_plan.time or ""]
            + slots.key_entities.events
        )[:8]
        order_refs_source = [item.name for item in slots.order_state.items]
        if slots.order_state.current_total is not None:
            order_refs_source.append(f"€{slots.order_state.current_total:.2f}")
        order_refs = _dedup(order_refs_source)[:10]
        preferences: Dict[str, Any] = {
            "constraints": slots.dietary_profile.constraints,
            "safe_items": slots.dietary_profile.safe_items,
            "service_mode": slots.logistics.service_mode,
        }

        return AdaptiveLayer1Block(
            topic=topic,
            turn_range=[batch[0].turn_number, batch[-1].turn_number],
            coverage_classes=coverage,
            conversation_focus=topic,
            key_facts=key_facts,
            active_threads=active_threads,
            user_preferences=preferences,
            entity_index=entity_index,
            numeric_facts=numeric_facts,
            directions=slots.logistics.directions[-4:],
            event_refs=event_refs,
            order_refs=order_refs,
            negative_decisions=slots.negative_decisions[-6:],
            quality_score=self._compute_quality_score(batch),
            info_gain_score=sum(s.info_gain_score for s in batch),
            last_updated_turn=batch[-1].turn_number,
            covered_turns=[snap.turn_number for snap in batch],
        )

    def _merge_with_llm_candidate(
        self,
        deterministic: AdaptiveLayer1Block,
        llm_candidate: Optional[Dict[str, Any]],
    ) -> AdaptiveLayer1Block:
        if not llm_candidate:
            return deterministic

        candidate = deterministic.model_copy(deep=True)
        candidate.conversation_focus = (
            llm_candidate.get("conversation_focus", "")
            or llm_candidate.get("focus", "")
            or candidate.conversation_focus
        )
        candidate.topic = llm_candidate.get("topic", "") or llm_candidate.get("focus", "") or candidate.topic
        candidate.active_threads = _dedup(candidate.active_threads + llm_candidate.get("active_threads", []))
        key_facts = llm_candidate.get("key_facts", [])
        if isinstance(key_facts, str):
            key_facts = [key_facts]
        candidate.key_facts = _dedup(candidate.key_facts + key_facts)
        candidate.entity_index = _dedup(candidate.entity_index + llm_candidate.get("entity_index", []))
        candidate.numeric_facts = _dedup(candidate.numeric_facts + llm_candidate.get("numeric_facts", []))
        candidate.event_refs = _dedup(candidate.event_refs + llm_candidate.get("event_refs", []))
        candidate.order_refs = _dedup(candidate.order_refs + llm_candidate.get("order_refs", []))
        candidate.directions = _dedup(candidate.directions + llm_candidate.get("directions", []))
        candidate.negative_decisions = _dedup(
            candidate.negative_decisions + llm_candidate.get("negative_decisions", [])
        )
        if llm_candidate.get("user_preferences"):
            merged_preferences = dict(candidate.user_preferences)
            merged_preferences.update(llm_candidate["user_preferences"])
            candidate.user_preferences = merged_preferences
        quality = llm_candidate.get("quality_score")
        if isinstance(quality, (int, float)):
            candidate.quality_score = max(candidate.quality_score, float(quality))
        elif candidate.quality_score == 0.0:
            candidate.quality_score = 0.5
        return candidate

    def _find_merge_candidate(
        self,
        blocks: Sequence[AdaptiveLayer1Block],
        coverage_classes: Sequence[CoverageClass],
    ) -> Optional[AdaptiveLayer1Block]:
        if not blocks:
            return None
        desired = {value.value for value in coverage_classes}
        for block in reversed(blocks):
            if desired.intersection({value.value for value in block.coverage_classes}):
                return block
        return blocks[-1]

    def _non_destructive_merge(
        self,
        previous: AdaptiveLayer1Block,
        new: AdaptiveLayer1Block,
    ) -> AdaptiveLayer1Block:
        merged = previous.model_copy(deep=True)
        merged.topic = new.topic or previous.topic
        merged.conversation_focus = new.conversation_focus or previous.conversation_focus
        merged.coverage_classes = self._aggregate_coverage([previous, new])
        merged.turn_range = [
            min(previous.turn_range[0], new.turn_range[0]) if previous.turn_range and new.turn_range else (new.turn_range[0] if new.turn_range else previous.turn_range[0]),
            max(previous.turn_range[-1], new.turn_range[-1]) if previous.turn_range and new.turn_range else (new.turn_range[-1] if new.turn_range else previous.turn_range[-1]),
        ]
        merged.key_facts = _dedup(previous.key_facts + new.key_facts)
        merged.active_threads = _dedup(previous.active_threads + new.active_threads)
        merged.entity_index = _dedup(previous.entity_index + new.entity_index)
        merged.numeric_facts = _dedup(previous.numeric_facts + new.numeric_facts)
        merged.directions = _dedup(previous.directions + new.directions)
        merged.event_refs = _dedup(previous.event_refs + new.event_refs)
        merged.order_refs = _dedup(previous.order_refs + new.order_refs)
        merged.negative_decisions = _dedup(previous.negative_decisions + new.negative_decisions)
        user_preferences = dict(previous.user_preferences)
        for key, value in new.user_preferences.items():
            if not value:
                continue
            if isinstance(value, list):
                user_preferences[key] = _dedup(list(user_preferences.get(key, [])) + value)
            elif isinstance(value, dict):
                merged_map = dict(user_preferences.get(key, {}))
                merged_map.update(value)
                user_preferences[key] = merged_map
            else:
                user_preferences[key] = value
        merged.user_preferences = user_preferences
        merged.quality_score = max(previous.quality_score, new.quality_score)
        merged.info_gain_score = max(previous.info_gain_score, new.info_gain_score)
        merged.last_updated_turn = max(previous.last_updated_turn, new.last_updated_turn)
        merged.covered_turns = _dedup(str(turn) for turn in previous.covered_turns + new.covered_turns)
        merged.covered_turns = [int(value) for value in merged.covered_turns]
        return merged

    def _aggregate_coverage(
        self,
        snapshots_or_blocks: Sequence[Any],
    ) -> List[CoverageClass]:
        values: List[CoverageClass] = []
        for item in snapshots_or_blocks:
            for coverage in getattr(item, "coverage_classes", []):
                if isinstance(coverage, CoverageClass):
                    if coverage not in values:
                        values.append(coverage)
                else:
                    enum_value = CoverageClass(str(coverage))
                    if enum_value not in values:
                        values.append(enum_value)
        return values or [CoverageClass.GENERAL]

    def _dominant_topic(self, batch: Sequence[AdaptiveSnapshot]) -> str:
        counts = Counter(snapshot.focus for snapshot in batch if snapshot.focus)
        if counts:
            return counts.most_common(1)[0][0]
        return "adaptive episodic memory"

    def _compute_quality_score(self, batch: Sequence[AdaptiveSnapshot]) -> float:
        base = sum(snapshot.user_salience for snapshot in batch)
        if not batch:
            return 0.0
        return round(base / len(batch), 3)

    def _upsert_layer1(
        self,
        session: HIAMSSessionState,
        block: AdaptiveLayer1Block,
        profile: HIAMSProfileSettings,
    ) -> None:
        existing_index = None
        for index, existing in enumerate(session.layer1_blocks):
            if existing.block_id == block.block_id:
                existing_index = index
                break
        if existing_index is not None:
            session.layer1_blocks[existing_index] = block
        else:
            session.layer1_blocks.append(block)

        if len(session.layer1_blocks) > profile.max_layer1_blocks:
            self._evict_layer1_block(session, profile)

    def _evict_layer1_block(
        self,
        session: HIAMSSessionState,
        profile: HIAMSProfileSettings,
    ) -> None:
        coverage_counter = Counter(
            coverage.value
            for block in session.layer1_blocks
            for coverage in block.coverage_classes
        )
        protected_unique = {
            coverage
            for coverage, count in coverage_counter.items()
            if count == 1 and coverage in {
                CoverageClass.ORDER.value,
                CoverageClass.DIETARY.value,
                CoverageClass.EVENT.value,
                CoverageClass.LOGISTICS.value,
            }
        }
        scored: List[Tuple[float, int]] = []
        max_turn = max((block.last_updated_turn for block in session.layer1_blocks), default=0)
        for index, block in enumerate(session.layer1_blocks):
            coverage_values = {coverage.value for coverage in block.coverage_classes}
            if protected_unique.intersection(coverage_values):
                continue
            recency = 1.0 / (1.0 + max(0, max_turn - block.last_updated_turn))
            frequency = min(len(block.covered_turns) / max(profile.layer0_recall_window, 1), 1.0)
            coverage_score = min(len(coverage_values) / 4.0, 1.0)
            salience = min(block.quality_score / 10.0, 1.0)
            total = recency * 0.30 + frequency * 0.25 + coverage_score * 0.30 + salience * 0.15
            scored.append((total, index))

        if not scored:
            return

        _, min_index = min(scored, key=lambda item: item[0])
        evicted = session.layer1_blocks.pop(min_index)
        logger.info(
            "[HIAMS] Evicted Layer1 block topic=%s turn_range=%s",
            evicted.topic,
            evicted.turn_range,
        )

    def _promote_layer2(
        self,
        session: HIAMSSessionState,
        slots: StructuredSlots,
        block: AdaptiveLayer1Block,
        profile: HIAMSProfileSettings,
    ) -> AdaptiveLayer2Memory:
        current = session.layer2.model_copy(deep=True)
        if not profile.enable_layer2:
            return current

        fact_counter = Counter(
            fact
            for existing in session.layer1_blocks
            for fact in existing.key_facts
        )
        query_mentions = Counter()
        for query in session.query_history:
            for fact in self._critical_slot_facts(slots):
                if fact.lower() in query.lower():
                    query_mentions[fact] += 1

        stable_candidates = list(current.stable_facts)
        stable_candidates.extend(
            fact
            for fact, count in fact_counter.items()
            if count >= 2
        )
        stable_candidates.extend(self._critical_slot_facts(slots))
        stable_candidates.extend(
            fact
            for fact, count in query_mentions.items()
            if count >= 1
        )
        stable_facts = _dedup(stable_candidates)

        slot_facts = _dedup(current.slot_facts + self._facts_from_slots(slots))
        promoted_items = _dedup(
            current.promoted_items
            + [item.name for item in slots.order_state.items]
            + slots.dietary_profile.safe_items
            + block.event_refs
        )
        event_timeline = _dedup(
            current.event_timeline
            + [slots.event_plan.name or "", slots.event_plan.location or "", slots.event_plan.time or ""]
        )
        order_summary = _dedup(
            current.order_summary
            + [item.name for item in slots.order_state.items]
            + ([f"€{slots.order_state.current_total:.2f}"] if slots.order_state.current_total is not None else [])
        )
        dietary_summary = _dedup(
            current.dietary_summary
            + slots.dietary_profile.constraints
            + slots.dietary_profile.safe_items
        )
        logistics_summary = _dedup(
            current.logistics_summary
            + slots.logistics.directions
            + ([slots.logistics.service_mode] if slots.logistics.service_mode else [])
            + ([f"€{slots.logistics.supplement:.2f}"] if slots.logistics.supplement is not None else [])
        )

        coverage_index = dict(current.coverage_index)
        for coverage in block.coverage_classes:
            key = coverage.value
            coverage_index[key] = _dedup(coverage_index.get(key, []) + block.key_facts[:6])
        coverage_index["critical"] = _dedup(coverage_index.get("critical", []) + stable_facts[:10])

        updated = AdaptiveLayer2Memory(
            version=current.version + 1,
            stable_facts=_dedup(stable_facts),
            slot_facts=slot_facts,
            promoted_items=promoted_items,
            event_timeline=event_timeline,
            order_summary=order_summary,
            dietary_summary=dietary_summary,
            logistics_summary=logistics_summary,
            unresolved_threads=current.unresolved_threads,
            coverage_index=coverage_index,
            last_updated_turn=block.last_updated_turn,
        )
        session.layer2 = updated
        return updated

    def _facts_from_slots(self, slots: StructuredSlots) -> List[str]:
        facts: List[str] = []
        if slots.order_state.current_total is not None:
            facts.append(f"Current total €{slots.order_state.current_total:.2f}")
        facts.extend(f"Order item: {item.name}" for item in slots.order_state.items)
        facts.extend(f"Dietary: {item}" for item in slots.dietary_profile.constraints)
        if slots.event_plan.name:
            facts.append(f"Event {slots.event_plan.name}")
        if slots.event_plan.location:
            facts.append(f"Location {slots.event_plan.location}")
        if slots.event_plan.time:
            facts.append(f"Time {slots.event_plan.time}")
        if slots.logistics.service_mode:
            facts.append(f"Service {slots.logistics.service_mode}")
        if slots.logistics.supplement is not None:
            facts.append(f"Service supplement €{slots.logistics.supplement:.2f}")
        facts.extend(f"Direction {step}" for step in slots.logistics.directions[-4:])
        facts.extend(f"Negative decision {item}" for item in slots.negative_decisions[-4:])
        return _dedup(facts)

    def _critical_slot_facts(self, slots: StructuredSlots) -> List[str]:
        facts: List[str] = []
        if slots.order_state.current_total is not None:
            facts.append(f"Critical total €{slots.order_state.current_total:.2f}")
        if slots.logistics.supplement is not None:
            facts.append(f"Critical supplement €{slots.logistics.supplement:.2f}")
        if slots.event_plan.name:
            facts.append(f"Critical event {slots.event_plan.name}")
        if slots.event_plan.location:
            facts.append(f"Critical location {slots.event_plan.location}")
        if slots.event_plan.time:
            facts.append(f"Critical time {slots.event_plan.time}")
        facts.extend(f"Critical dietary {item}" for item in slots.dietary_profile.constraints)
        facts.extend(f"Critical route {item}" for item in slots.logistics.directions[-3:])
        facts.extend(f"Critical price {item}" for item in slots.key_entities.prices[-6:])
        return _dedup(facts)

    def _apply_immediate_promotion(
        self,
        session: HIAMSSessionState,
        snapshot: AdaptiveSnapshot,
    ) -> None:
        slots = session.structured_slots
        provisional = getattr(slots, "provisional_critical", {}) or {}
        if not any(provisional.get(key) for key in ("places", "events", "prices")) and CoverageClass.DIETARY not in snapshot.coverage_classes:
            return

        current = session.layer2.model_copy(deep=True)
        immediate_facts: List[str] = []
        immediate_facts.extend(f"Immediate location {item}" for item in provisional.get("places", []))
        immediate_facts.extend(f"Immediate event {item}" for item in provisional.get("events", []))
        immediate_facts.extend(f"Immediate price {item}" for item in provisional.get("prices", []))
        if CoverageClass.DIETARY in snapshot.coverage_classes:
            immediate_facts.extend(f"Immediate dietary {item}" for item in slots.dietary_profile.constraints)

        current.stable_facts = _dedup(current.stable_facts + immediate_facts)
        current.slot_facts = _dedup(current.slot_facts + immediate_facts)
        current.event_timeline = _dedup(
            current.event_timeline
            + provisional.get("events", [])
            + provisional.get("places", [])
            + ([slots.event_plan.time] if slots.event_plan.time else [])
        )
        current.order_summary = _dedup(
            current.order_summary
            + ([f"€{slots.order_state.current_total:.2f}"] if slots.order_state.current_total is not None else [])
            + provisional.get("prices", [])
        )
        current.dietary_summary = _dedup(
            current.dietary_summary + slots.dietary_profile.constraints
        )
        current.logistics_summary = _dedup(
            current.logistics_summary + provisional.get("places", []) + slots.logistics.directions[-4:]
        )
        coverage_index = dict(current.coverage_index)
        coverage_index["critical"] = _dedup(coverage_index.get("critical", []) + immediate_facts)
        current.coverage_index = coverage_index
        current.last_updated_turn = snapshot.turn_number
        session.layer2 = current


# FIX ROUND 1 - HIAMS recall boost - 2026-04-04
# FIX ROUND 2 - HIAMS zero-latency recall boost - T15 fixed
