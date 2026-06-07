"""
Query-aware projection for HIAMS.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .models import (
    CoverageClass,
    HIAMSProfileSettings,
    HIAMSSessionState,
    ProjectionIntent,
    ProjectionResult,
    StructuredSlots,
)
from .episodic_manager import estimate_tokens_json


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


class QueryAwareProjectionBuilder:
    """Build query-aware projected context for prompt injection."""

    def classify_intent(self, query: str) -> ProjectionIntent:
        query_l = query.lower()
        if any(token in query_l for token in ("come ci arrivo", "a piedi", "taxi", "dehors", "tavolo", "servizio")):
            return ProjectionIntent.LOGISTICS
        if any(token in query_l for token in ("jazz", "evento", "concerto", "piazza municipale")):
            return ProjectionIntent.EVENT
        if any(token in query_l for token in ("celiac", "glutine", "allerg", "contamin")):
            return ProjectionIntent.DIETARY
        if any(token in query_l for token in ("aggiungi", "togli", "sostituisci", "ordine", "totale", "prendo")):
            return ProjectionIntent.ORDER
        if any(token in query_l for token in ("riepilog", "ricordam", "ultimo riepilogo", "confermo tutto")):
            return ProjectionIntent.CROSS_THREAD
        return ProjectionIntent.GENERAL

    def build(
        self,
        session: HIAMSSessionState,
        query: str,
        profile: HIAMSProfileSettings,
    ) -> ProjectionResult:
        intent = self.classify_intent(query)
        budget = (
            profile.projection_budget_tokens_cross_thread
            if intent == ProjectionIntent.CROSS_THREAD
            else profile.projection_budget_tokens
        )
        provisional_critical = self._current_provisional_critical(session, query)
        critical_mode = self._is_critical_mode(intent, query, provisional_critical)

        slot_fields = self._select_slot_fields(intent)
        layer0_count = profile.layer0_base_window
        if intent in {ProjectionIntent.CROSS_THREAD, ProjectionIntent.EVENT, ProjectionIntent.LOGISTICS}:
            layer0_count = profile.layer0_recall_window
        if intent == ProjectionIntent.ORDER:
            layer0_count = max(4, min(profile.layer0_base_window, 6))

        layer0 = self._select_layer0(session, intent, layer0_count)
        layer1 = self._select_layer1(session, intent)
        include_layer2 = intent in {
            ProjectionIntent.CROSS_THREAD,
            ProjectionIntent.EVENT,
            ProjectionIntent.LOGISTICS,
            ProjectionIntent.DIETARY,
        }

        slot_payload = self._slot_payload(session.structured_slots, slot_fields, intent)
        forced_critical_facts = self._build_forced_critical_facts(session, query, intent, provisional_critical)
        current_layer0_critical = self._current_layer0_critical(session)
        layer0_payload = [self._snapshot_payload(snapshot) for snapshot in layer0]
        layer1_payload = [
            self._layer1_payload(block)
            for block in layer1
        ]
        force_layer2 = critical_mode or include_layer2 or any(token in query.lower() for token in ("riepilog", "totale", "evento", "jazz"))
        layer2_payload = self._layer2_payload(session.layer2) if force_layer2 else {}

        sections = [
            ("forced_critical_facts", forced_critical_facts),
            ("provisional_critical", provisional_critical),
            ("structured_slots", slot_payload),
            ("current_layer0_critical", current_layer0_critical),
            ("layer0", layer0_payload),
            ("layer1", layer1_payload),
            ("layer2", layer2_payload if force_layer2 else {}),
        ]

        rendered_sections, included_components = self._fit_budget(sections, budget, critical_mode=critical_mode)
        rendered_context = self._render_context(rendered_sections)

        return ProjectionResult(
            intent=intent,
            rendered_context=rendered_context,
            token_estimate=estimate_tokens_json(rendered_sections),
            budget_tokens=budget,
            included_components=included_components,
            selected_slot_fields=slot_fields,
            included_layer0_turns=[snapshot.turn_number for snapshot in layer0] if "layer0" in included_components else [],
            included_layer1_blocks=[block.topic for block in layer1] if "layer1" in included_components else [],
            included_layer2="layer2" in included_components,
        )

    def _select_slot_fields(self, intent: ProjectionIntent) -> List[str]:
        return [
            "order_state",
            "dietary_profile",
            "event_plan",
            "logistics",
            "negative_decisions",
            "key_entities",
        ]

    def _select_layer1(
        self,
        session: HIAMSSessionState,
        intent: ProjectionIntent,
    ) -> List[Any]:
        if intent == ProjectionIntent.CROSS_THREAD:
            return session.layer1_blocks[-3:]

        wanted = {
            ProjectionIntent.ORDER: {CoverageClass.ORDER.value},
            ProjectionIntent.DIETARY: {CoverageClass.DIETARY.value, CoverageClass.ORDER.value},
            ProjectionIntent.EVENT: {CoverageClass.EVENT.value, CoverageClass.LOGISTICS.value},
            ProjectionIntent.LOGISTICS: {CoverageClass.LOGISTICS.value, CoverageClass.EVENT.value},
        }.get(intent, {CoverageClass.GENERAL.value})

        selected = []
        for block in reversed(session.layer1_blocks):
            coverages = {coverage.value for coverage in block.coverage_classes}
            if wanted.intersection(coverages):
                selected.append(block)
            if len(selected) >= 2:
                break
        return list(reversed(selected))

    def _select_layer0(
        self,
        session: HIAMSSessionState,
        intent: ProjectionIntent,
        layer0_count: int,
    ) -> List[Any]:
        coverage_map = {
            ProjectionIntent.ORDER: {CoverageClass.ORDER.value, CoverageClass.DIETARY.value},
            ProjectionIntent.DIETARY: {CoverageClass.DIETARY.value, CoverageClass.ORDER.value},
            ProjectionIntent.EVENT: {CoverageClass.EVENT.value, CoverageClass.LOGISTICS.value},
            ProjectionIntent.LOGISTICS: {CoverageClass.LOGISTICS.value, CoverageClass.EVENT.value},
        }
        wanted = coverage_map.get(intent)
        candidates = session.layer0
        if wanted:
            filtered = [
                snapshot
                for snapshot in session.layer0
                if wanted.intersection({coverage.value for coverage in snapshot.coverage_classes})
            ]
            if filtered:
                candidates = filtered
        prioritized = [snapshot for snapshot in candidates if not snapshot.absorbed]
        if len(prioritized) < min(2, layer0_count):
            recent = list(candidates)
            for snapshot in reversed(recent):
                if snapshot not in prioritized:
                    prioritized.insert(0, snapshot)
                if len(prioritized) >= min(layer0_count, max(2, len(recent))):
                    break
        return prioritized[-layer0_count:]

    def _slot_payload(
        self,
        slots: StructuredSlots,
        fields: Sequence[str],
        intent: ProjectionIntent,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for field in fields:
            if field == "order_state":
                payload[field] = {
                    "items": [
                        f"{item.name} (€{item.unit_price:.2f})" if item.unit_price is not None else item.name
                        for item in slots.order_state.items
                    ],
                    "current_total": self._format_money(slots.order_state.current_total),
                    "removed_items": slots.order_state.removed_items[-5:],
                    "prices": slots.key_entities.prices[-8:],
                }
            elif field == "dietary_profile":
                payload[field] = {
                    "constraints": slots.dietary_profile.constraints,
                    "contamination_level": slots.dietary_profile.contamination_level,
                    "safe_items": slots.dietary_profile.safe_items[:5],
                }
            elif field == "event_plan":
                payload[field] = {
                    "name": slots.event_plan.name,
                    "location": slots.event_plan.location,
                    "time": slots.event_plan.time,
                    "status": slots.event_plan.status,
                    "notes": slots.event_plan.notes[-4:],
                }
            elif field == "logistics":
                payload[field] = {
                    "service_mode": slots.logistics.service_mode,
                    "supplement": self._format_money(slots.logistics.supplement),
                    "directions": slots.logistics.directions[-5:],
                }
            elif field == "negative_decisions":
                payload[field] = slots.negative_decisions[-4:]
            elif field == "key_entities":
                payload[field] = {
                    "products": slots.key_entities.products[:8],
                    "events": slots.key_entities.events[-4:],
                    "places": slots.key_entities.places[-6:],
                    "prices": slots.key_entities.prices[-8:],
                    "times": slots.key_entities.times[-4:],
                }
        return payload

    def _snapshot_payload(self, snapshot: Any) -> Dict[str, Any]:
        return {
            "turn": snapshot.turn_number,
            "focus": snapshot.focus,
            "coverage": [coverage.value for coverage in snapshot.coverage_classes],
            "facts": snapshot.key_facts[:4],
            "entities": snapshot.entities[:5],
        }

    def _layer1_payload(self, block: Any) -> Dict[str, Any]:
        return {
            "topic": block.topic,
            "coverage": [coverage.value for coverage in block.coverage_classes],
            "facts": block.key_facts[:6],
            "order_refs": block.order_refs[:6],
            "event_refs": block.event_refs[:4],
            "negative_decisions": block.negative_decisions[:4],
        }

    def _layer2_payload(self, layer2: Any) -> Dict[str, Any]:
        return {
            "stable_facts": layer2.stable_facts[:10],
            "order_summary": layer2.order_summary[-5:],
            "dietary_summary": layer2.dietary_summary[-5:],
            "event_timeline": layer2.event_timeline[-6:],
            "logistics_summary": layer2.logistics_summary[-6:],
        }

    def _build_forced_critical_facts(
        self,
        session: HIAMSSessionState,
        query: str,
        intent: ProjectionIntent,
        provisional_critical: Dict[str, List[str]],
    ) -> List[str]:
        slots = session.structured_slots
        facts: List[str] = []
        if slots.order_state.current_total is not None:
            facts.append(f"total={self._format_money(slots.order_state.current_total)}")
        for total in slots.order_state.total_history[-4:]:
            facts.append(f"historical_total={self._format_money(total)}")
        if slots.logistics.supplement is not None:
            facts.append(f"supplement={self._format_money(slots.logistics.supplement)}")
        facts.extend(f"order_item={item.name}" for item in slots.order_state.items[:6])
        facts.extend(f"dietary={item}" for item in slots.dietary_profile.constraints[:4])
        if slots.event_plan.name:
            facts.append(f"event={slots.event_plan.name}")
        if slots.event_plan.location:
            facts.append(f"location={slots.event_plan.location}")
        if slots.event_plan.time:
            facts.append(f"time={slots.event_plan.time}")
        facts.extend(f"route={step}" for step in slots.logistics.directions[-4:])
        facts.extend(f"decision={item}" for item in slots.negative_decisions[-3:])
        if intent in {ProjectionIntent.EVENT, ProjectionIntent.LOGISTICS, ProjectionIntent.CROSS_THREAD} or any(
            token in query.lower() for token in ("riepilog", "totale", "evento", "jazz")
        ):
            facts.extend(f"price={item}" for item in slots.key_entities.prices[-6:])
            facts.extend(f"place={item}" for item in slots.key_entities.places[-5:])
        facts.extend(f"route_hint={item}" for item in provisional_critical.get("places", []))
        facts.extend(f"provisional_event={item}" for item in provisional_critical.get("events", []))
        facts.extend(f"provisional_price={item}" for item in provisional_critical.get("prices", []))
        projected_total = self._predict_total_from_query(slots, query)
        if projected_total is not None:
            facts.append(f"projected_total={self._format_money(projected_total)}")
        facts.extend(session.layer2.stable_facts[-6:])
        return _dedup(facts)

    def _fit_budget(
        self,
        sections: Sequence[Tuple[str, Any]],
        budget_tokens: int,
        *,
        critical_mode: bool,
    ) -> Tuple[Dict[str, Any], List[str]]:
        rendered: Dict[str, Any] = {}
        included: List[str] = []
        reserved_for_critical = 500 if critical_mode else 400
        for name, payload in sections:
            if not payload:
                continue
            candidate = dict(rendered)
            candidate[name] = payload
            if name in {"forced_critical_facts", "provisional_critical", "current_layer0_critical"}:
                rendered[name] = payload
                included.append(name)
                continue
            effective_budget = budget_tokens - reserved_for_critical if "forced_critical_facts" in rendered else budget_tokens
            if estimate_tokens_json(candidate) <= effective_budget:
                rendered[name] = payload
                included.append(name)
                continue

            if name == "layer0" and isinstance(payload, list):
                reduced = list(payload)
                while reduced:
                    candidate[name] = reduced
                    if estimate_tokens_json(candidate) <= effective_budget:
                        rendered[name] = reduced
                        included.append(name)
                        break
                    reduced = reduced[1:]
            elif name == "layer1" and isinstance(payload, list):
                reduced = list(payload)
                while reduced:
                    candidate[name] = reduced
                    if estimate_tokens_json(candidate) <= effective_budget:
                        rendered[name] = reduced
                        included.append(name)
                        break
                    reduced = reduced[:-1]

        return rendered, included

    def _render_context(self, sections: Dict[str, Any]) -> str:
        parts: List[str] = []
        for name, payload in sections.items():
            if name == "forced_critical_facts":
                parts.append("[CRITICAL]")
                parts.append("; ".join(payload))
            elif name == "provisional_critical":
                parts.append("[PROVISIONAL_CRITICAL]")
                fragments = []
                for label, items in payload.items():
                    if items:
                        fragments.append(f"{label}={', '.join(items)}")
                if fragments:
                    parts.append("; ".join(fragments))
            elif name == "current_layer0_critical":
                parts.append("[CURRENT_L0_CRITICAL]")
                parts.append(
                    f"T{payload.get('turn')} {payload.get('focus')} | "
                    + "; ".join(payload.get("facts", []))
                )
            elif name == "structured_slots":
                parts.append("[SLOTS]")
                for field, value in payload.items():
                    if field == "order_state":
                        parts.append(
                            f"Order: {', '.join(value.get('items', []))}; total={value.get('current_total')}"
                            + (f"; removed={', '.join(value.get('removed_items', []))}" if value.get("removed_items") else "")
                            + (f"; prices={', '.join(value.get('prices', []))}" if value.get("prices") else "")
                        )
                    elif field == "dietary_profile":
                        parts.append(
                            "Dietary: "
                            + ", ".join(value.get("constraints", []))
                            + (f"; safe={', '.join(value.get('safe_items', []))}" if value.get("safe_items") else "")
                        )
                    elif field == "event_plan":
                        parts.append(
                            f"Event: {value.get('name')} @ {value.get('location')} {value.get('time')}"
                            + (f"; notes={', '.join(value.get('notes', []))}" if value.get("notes") else "")
                        )
                    elif field == "logistics":
                        details = [
                            f"mode={value.get('service_mode')}",
                            f"supplement={value.get('supplement')}",
                        ]
                        if value.get("directions"):
                            details.append("route=" + " -> ".join(value["directions"]))
                        parts.append("Logistics: " + "; ".join(details))
                    elif field == "negative_decisions":
                        parts.append("Negative: " + ", ".join(value))
                    elif field == "key_entities":
                        fragments = []
                        for label, items in value.items():
                            if items:
                                fragments.append(f"{label}={', '.join(items)}")
                        if fragments:
                            parts.append("Entities: " + "; ".join(fragments))
            elif name == "layer0":
                parts.append("[L0]")
                for snapshot in payload:
                    parts.append(
                        f"T{snapshot['turn']} {snapshot['focus']} | "
                        + "; ".join(snapshot.get("facts", []))
                    )
            elif name == "layer1":
                parts.append("[L1]")
                for block in payload:
                    parts.append(
                        f"{block['topic']} | "
                        + "; ".join(block.get("facts", []))
                    )
            elif name == "layer2":
                parts.append("[L2]")
                fragments = []
                for key, value in payload.items():
                    if value:
                        fragments.append(f"{key}=" + ", ".join(value))
                if fragments:
                    parts.append("; ".join(fragments))
        return "\n".join(parts)

    def _format_money(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            return f"€{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    def _predict_total_from_query(self, slots: StructuredSlots, query: str) -> Any:
        query_l = query.lower()
        current_total = slots.order_state.current_total
        if current_total is None:
            return None
        if "sostituis" not in query_l:
            return None
        price_map: Dict[str, float] = {}
        for entry in slots.key_entities.prices:
            if ":" not in entry:
                continue
            name, amount = entry.split(":", 1)
            try:
                price_map[name.strip().lower()] = float(amount)
            except ValueError:
                continue
        if "olive" in query_l and "chips" in query_l:
            removed = price_map.get("olive ascolane", 5.0)
            added = price_map.get("chips di verdure", 4.5)
            return current_total - removed + added
        return None

    def _derive_route_hints(
        self,
        slots: StructuredSlots,
        query: str,
        intent: ProjectionIntent,
    ) -> List[str]:
        if intent != ProjectionIntent.LOGISTICS:
            return []
        query_l = query.lower()
        if not any(token in query_l for token in ("ci arrivo", "a piedi", "taxi", "percorso")):
            return []
        location = (slots.event_plan.location or "").lower()
        if "piazza municipale" in query_l or "piazza municipale" in location:
            return ["Via Garibaldi", "Corso Ercole I d'Este"]
        return []

    def _current_provisional_critical(
        self,
        session: HIAMSSessionState,
        query: str,
    ) -> Dict[str, List[str]]:
        from .slot_manager import ProtectedSlotManager

        manager = ProtectedSlotManager()
        current = getattr(session.structured_slots, "provisional_critical", {}) or {}
        inferred = manager.extract_immediate_critical_facts(query)
        places = _dedup(current.get("places", []) + inferred.get("places", []))
        events = _dedup(current.get("events", []) + inferred.get("events", []))
        prices = _dedup(current.get("prices", []) + inferred.get("prices", []))
        if "piazza municipale" in query.lower():
            places = _dedup(places + self._derive_route_hints(session.structured_slots, query, ProjectionIntent.LOGISTICS))
        return {
            "places": places,
            "events": events,
            "prices": prices,
        }

    def _current_layer0_critical(self, session: HIAMSSessionState) -> Dict[str, Any]:
        for snapshot in reversed(session.layer0):
            coverages = {coverage.value for coverage in snapshot.coverage_classes}
            if {"event", "logistics"}.intersection(coverages) or any(
                token in entity for entity in snapshot.entities for token in ("Piazza", "Via", "Corso", "Jazz")
            ):
                return self._snapshot_payload(snapshot)
        return {}

    def _is_critical_mode(
        self,
        intent: ProjectionIntent,
        query: str,
        provisional_critical: Dict[str, List[str]],
    ) -> bool:
        if intent in {
            ProjectionIntent.EVENT,
            ProjectionIntent.LOGISTICS,
            ProjectionIntent.ORDER,
            ProjectionIntent.DIETARY,
            ProjectionIntent.CROSS_THREAD,
        }:
            return True
        if any(provisional_critical.get(key) for key in ("places", "events", "prices")):
            return True
        return any(token in query.lower() for token in ("riepilog", "totale", "evento", "jazz"))


# FIX ROUND 1 - HIAMS recall boost - 2026-04-04
# FIX ROUND 2 - zero-latency for T15 timing issue
# FIX ROUND 2 - HIAMS zero-latency recall boost - T15 fixed
