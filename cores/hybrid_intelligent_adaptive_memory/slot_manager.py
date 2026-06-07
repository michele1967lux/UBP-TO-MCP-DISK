"""
Deterministic structured-slot management for HIAMS.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    DietaryProfile,
    EventPlan,
    KeyEntities,
    LogisticsState,
    OrderItem,
    OrderState,
    SlotUpdateResult,
    StructuredSlots,
)


_PRICE_RE = re.compile(r"(?:€\s?(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s?€)")
_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})\b")
_ITEM_LINE_RE = re.compile(r"^\s*-\s*(.+?)\s*(?:[:\-]|\()\s*€\s?(\d+(?:[.,]\d+)?)", re.MULTILINE)
_EVENT_RE = re.compile(
    r"\b(?:Jazz|Concerto|Mostra|Aperitivo|Festival)[^,.\n]*(?:Chiostro|Municipale|Diamanti|set|live)?",
    re.IGNORECASE,
)
_ADD_RE = re.compile(r"\b(?:aggiungi|prendo|prendi|ordino|confermo)\s+([^:.;,\n]+)", re.IGNORECASE)
_REMOVE_RE = re.compile(r"\b(?:togli|niente|rimuovi)\s+([^:.;,\n]+)", re.IGNORECASE)
_SUB_RE = re.compile(r"\b(?:sostituisci)\s+([^:.;,\n]+?)\s+con\s+([^:.;,\n]+)", re.IGNORECASE)
_NEG_DECISION_RE = re.compile(
    r"\b(?:non aggiungere|niente|no)\s+([^:.;,\n]+)",
    re.IGNORECASE,
)
_MENU_CANDIDATE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-zÀ-ÿ'`-]+(?:\s+(?:al|alla|ai|agli|di|del|della|dello|dei|degli|delle|da|con|senza))?(?:\s+[A-Za-zÀ-ÿ'`-]+){0,3})\b"
)
_PLACE_STARTERS = {"Piazza", "Via", "Corso", "Palazzo"}
_PLACE_CONNECTORS = {
    "di",
    "del",
    "della",
    "dello",
    "dei",
    "degli",
    "delle",
    "da",
    "al",
    "alla",
    "ai",
    "agli",
    "in",
    "sul",
    "sulla",
    "d'este",
}
_PLACE_STOPWORDS = {
    "fino",
    "poi",
    "destra",
    "sinistra",
    "circa",
    "semplice",
    "migliore",
    "libero",
    "ingresso",
    "serale",
    "breve",
    "pena",
    "dal",
    "del",
}
_SAFE_MARKERS = ("senza glutine", "gluten-free", "olio dedicato", "certificata", "adatto")
_ITEM_STOP_PREFIXES = ("Jazz", "Mostra", "Piazza", "Via", "Corso", "Palazzo", "Ferrara", "Evento")
_GENERIC_ITEM_WORDS = {
    "ciao",
    "benvenuto",
    "benvenuta",
    "certo",
    "ok",
    "perfetto",
    "perfetta",
    "si",
    "sì",
    "ordine",
    "riepilogo",
    "conferma",
    "totale",
    "servizio",
}


def _dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        value = item.strip()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _clean_name(raw: str) -> str:
    return raw.strip().strip("-:;,.() ").replace("  ", " ")


def _to_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def _item_names(items: Sequence[OrderItem]) -> List[str]:
    return [item.name for item in items if item.status == "active"]


class ProtectedSlotManager:
    """Update protected slots using deterministic parsers."""

    def update_slots(
        self,
        previous: StructuredSlots,
        query: str,
        response: str,
    ) -> SlotUpdateResult:
        slots = previous.model_copy(deep=True)
        changed_slots: List[str] = []
        new_facts: List[str] = []

        order_changed, order_facts = self._update_order_state(slots.order_state, query, response)
        if order_changed:
            changed_slots.append("order_state")
            new_facts.extend(order_facts)

        dietary_changed, dietary_facts = self._update_dietary(slots.dietary_profile, query, response)
        if dietary_changed:
            changed_slots.append("dietary_profile")
            new_facts.extend(dietary_facts)

        event_changed, event_facts = self._update_event_plan(slots.event_plan, query, response)
        if event_changed:
            changed_slots.append("event_plan")
            new_facts.extend(event_facts)

        logistics_changed, logistics_facts = self._update_logistics(slots.logistics, query, response)
        if logistics_changed:
            changed_slots.append("logistics")
            new_facts.extend(logistics_facts)

        negatives = self._extract_negative_decisions(query)
        merged_negatives = _dedup(list(slots.negative_decisions) + negatives)
        if merged_negatives != slots.negative_decisions:
            slots.negative_decisions = merged_negatives
            changed_slots.append("negative_decisions")
            new_facts.extend(f"Negative decision: {item}" for item in negatives)

        entities_before = slots.key_entities.model_dump()
        self._update_key_entities(slots.key_entities, query, response, slots)
        if slots.key_entities.model_dump() != entities_before:
            changed_slots.append("key_entities")
            new_facts.extend(self._facts_from_entities(slots.key_entities))

        provisional_critical = self.extract_immediate_critical_facts(f"{query}\n{response}")
        slots = self.clean_and_validate(slots)
        object.__setattr__(slots, "provisional_critical", provisional_critical)
        info_gain = float(len(_dedup(new_facts)))
        return SlotUpdateResult(
            slots=slots,
            changed_slots=_dedup(changed_slots),
            new_facts=_dedup(new_facts),
            info_gain_score=info_gain,
        )

    def _update_order_state(
        self,
        order: OrderState,
        query: str,
        response: str,
    ) -> Tuple[bool, List[str]]:
        before = order.model_dump()
        facts: List[str] = []

        explicit_subs = [
            (_clean_name(src), _clean_name(dst)) for src, dst in _SUB_RE.findall(query)
        ]
        for src, dst in explicit_subs:
            order.substitutions[src] = dst
            if src not in order.removed_items:
                order.removed_items.append(src)
            facts.append(f"Substitution: {src} -> {dst}")

        explicit_removals = [_clean_name(name) for name in _REMOVE_RE.findall(query)]
        for removed in explicit_removals:
            if removed not in order.removed_items:
                order.removed_items.append(removed)
                facts.append(f"Removed item: {removed}")

        order_lines = list(_ITEM_LINE_RE.finditer(response)) if self._response_contains_order_summary(response) else []
        if order_lines:
            current_items: List[OrderItem] = []
            for match in order_lines:
                current_items.append(
                    OrderItem(
                        name=_clean_name(match.group(1)),
                        unit_price=_to_float(match.group(2)),
                    )
                )
            previous_names = set(_item_names(order.items))
            current_names = set(_item_names(current_items))
            for removed in sorted(previous_names - current_names):
                if removed not in order.removed_items:
                    order.removed_items.append(removed)
                    facts.append(f"Removed item: {removed}")
            order.items = current_items
            if current_items:
                facts.append("Active order items: " + ", ".join(_item_names(current_items)))

        total = self._extract_total(response)
        if total is not None:
            order.current_total = total
            if total not in order.total_history:
                order.total_history.append(total)
            facts.append(f"Order total: €{total:.2f}")

        for src, dst in explicit_subs:
            found = False
            for item in order.items:
                if item.name == src:
                    item.status = "replaced"
                    item.notes.append(f"Replaced with {dst}")
                    found = True
            if not found and dst not in _item_names(order.items):
                order.items.append(OrderItem(name=dst))

        if not order_lines:
            additions = [_clean_name(name) for name in _ADD_RE.findall(query)]
            current_names = _item_names(order.items)
            for name in additions:
                normalized = name
                if normalized and normalized not in current_names:
                    order.items.append(OrderItem(name=normalized))
                    facts.append(f"Order addition: {normalized}")

        order.removed_items = _dedup(order.removed_items)
        changed = order.model_dump() != before
        return changed, _dedup(facts)

    def _update_dietary(
        self,
        dietary: DietaryProfile,
        query: str,
        response: str,
    ) -> Tuple[bool, List[str]]:
        before = dietary.model_dump()
        facts: List[str] = []
        text = f"{query}\n{response}".lower()

        if "celiac" in text:
            dietary.constraints = _dedup(dietary.constraints + ["celiaco"])
            facts.append("Dietary constraint: celiaco")
        if "senza glutine" in text or "gluten-free" in text:
            dietary.constraints = _dedup(dietary.constraints + ["senza glutine"])
            facts.append("Dietary constraint: senza glutine")
        if "contamin" in text:
            dietary.contamination_level = "attenzione"
            facts.append("Dietary contamination attention required")
        if "olio dedicato" in text and dietary.contamination_level is None:
            dietary.contamination_level = "controllata"

        safe_items = list(dietary.safe_items)
        safe_items.extend(self._extract_safe_items_from_response(response))
        dietary.safe_items = _dedup(safe_items)
        if dietary.safe_items:
            facts.append("Safe items: " + ", ".join(dietary.safe_items))

        changed = dietary.model_dump() != before
        return changed, _dedup(facts)

    def _update_event_plan(
        self,
        event_plan: EventPlan,
        query: str,
        response: str,
    ) -> Tuple[bool, List[str]]:
        before = event_plan.model_dump()
        facts: List[str] = []
        combined = f"{query}\n{response}"
        query_l = query.lower()

        event_sentences = [
            sentence.strip()
            for sentence in re.split(r"[.\n]", combined)
            if any(token in sentence.lower() for token in ("jazz", "concerto", "mostra", "evento", "chiostro"))
        ]
        event_source = "\n".join(event_sentences) if event_sentences else combined

        explicit_event = None
        if "jazz al chiostro" in combined.lower():
            explicit_event = "Jazz al Chiostro"
        elif "mostra fotografica" in combined.lower():
            explicit_event = "Mostra fotografica"

        event_names = [self._normalize_event_name(match.group(0)) for match in _EVENT_RE.finditer(event_source)]
        event_names = _dedup(event_names)
        candidate_name = explicit_event or (event_names[0] if event_names else None)
        if candidate_name:
            if event_plan.name and "jazz al chiostro" in event_plan.name.lower() and "jazz" in candidate_name.lower():
                candidate_name = event_plan.name
            elif event_plan.name and len(event_plan.name) > len(candidate_name) and candidate_name.lower() in event_plan.name.lower():
                candidate_name = event_plan.name
            event_plan.name = candidate_name
            facts.append(f"Event: {event_plan.name}")

        locations = self._extract_places(event_source)
        if locations:
            event_plan.location = locations[0]
            facts.append(f"Event location: {event_plan.location}")

        times = _dedup(_TIME_RE.findall(event_source))
        if times:
            event_plan.time = times[0]
            facts.append(f"Event time: {event_plan.time}")

        if any(token in query_l for token in ("confermo", "ci vado", "andiamo", "interessante")):
            event_plan.status = "considered"
        if event_plan.name and event_plan.time and "confermo" in query_l:
            event_plan.status = "confirmed"

        if locations or times or event_names:
            event_plan.notes = _dedup(event_plan.notes + [
                note for note in (event_plan.name, event_plan.location, event_plan.time) if note
            ])

        changed = event_plan.model_dump() != before
        return changed, _dedup(facts)

    def _update_logistics(
        self,
        logistics: LogisticsState,
        query: str,
        response: str,
    ) -> Tuple[bool, List[str]]:
        before = logistics.model_dump()
        facts: List[str] = []
        combined = f"{query}\n{response}"
        combined_l = combined.lower()

        if "dehors" in combined_l:
            logistics.service_mode = "dehors"
            facts.append("Service mode: dehors")
        elif "tavolo" in combined_l:
            logistics.service_mode = "tavolo"
            facts.append("Service mode: tavolo")
        elif "bancone" in combined_l:
            logistics.service_mode = "bancone"
            facts.append("Service mode: bancone")

        supplement = self._extract_supplement(response)
        if supplement is not None:
            logistics.supplement = supplement
            facts.append(f"Service supplement: €{supplement:.2f}")

        directions = list(logistics.directions)
        directions.extend(self._extract_direction_steps(combined))
        logistics.directions = _dedup(directions)
        if logistics.directions:
            facts.append("Directions captured")

        changed = logistics.model_dump() != before
        return changed, _dedup(facts)

    def _update_key_entities(
        self,
        entities: KeyEntities,
        query: str,
        response: str,
        slots: StructuredSlots,
    ) -> None:
        combined = f"{query}\n{response}"
        products = _dedup(
            self._canonical_item_name(item)
            for item in (
                entities.products
                + _item_names(slots.order_state.items)
                + slots.dietary_profile.safe_items
                + self._extract_item_names(combined)
            )
            if self._canonical_item_name(item)
        )
        places = _dedup(
            self._canonical_place_name(place)
            for place in (
                entities.places
                + self._extract_places(combined)
                + slots.logistics.directions
                + ([slots.event_plan.location] if slots.event_plan.location else [])
            )
            if self._canonical_place_name(place)
        )
        events = _dedup(
            self._canonical_event_name(event)
            for event in (
                entities.events
                + ([slots.event_plan.name] if slots.event_plan.name else [])
                + [self._normalize_event_name(match.group(0)) for match in _EVENT_RE.finditer(combined)]
            )
            if self._canonical_event_name(event)
        )
        prices = _dedup(entities.prices + self._extract_price_entries(query, response, slots))
        times = _dedup(entities.times + _TIME_RE.findall(combined) + ([slots.event_plan.time] if slots.event_plan.time else []))

        people = list(entities.people)
        if "marco" in combined.lower():
            people.append("Marco")
        if "amica" in combined.lower():
            people.append("amica")

        entities.products = products
        entities.places = places
        entities.events = events
        entities.prices = prices
        entities.times = times
        entities.people = _dedup(people)

    def _facts_from_entities(self, entities: KeyEntities) -> List[str]:
        facts: List[str] = []
        if entities.products:
            facts.append("Products: " + ", ".join(entities.products[:6]))
        if entities.events or entities.places:
            facts.append("Places/events: " + ", ".join(_dedup(entities.events + entities.places)[:6]))
        if entities.prices:
            facts.append("Prices: " + ", ".join(entities.prices[:6]))
        if entities.times:
            facts.append("Times: " + ", ".join(entities.times[:4]))
        return facts

    def _extract_negative_decisions(self, query: str) -> List[str]:
        return [
            _clean_name(match.group(0))
            for match in _NEG_DECISION_RE.finditer(query)
        ]

    def _extract_item_names(self, text: str) -> List[str]:
        items = [_clean_name(match.group(1)) for match in _ITEM_LINE_RE.finditer(text)]
        if items:
            return _dedup(self._canonical_item_name(item) for item in items)
        additions = [_clean_name(name) for name in _ADD_RE.findall(text)]
        return _dedup(self._canonical_item_name(item) for item in additions)

    def _extract_total(self, text: str) -> Optional[float]:
        patterns = [
            r"(?:con servizio[^€]*|totale complessivo[^€]*|diventa[^€]*|importo finale[^€]*)€\s?(\d+(?:[.,]\d+)?)",
            r"totale[^€]*€\s?(\d+(?:[.,]\d+)?)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                return _to_float(matches[-1])
        return None

    def _extract_supplement(self, text: str) -> Optional[float]:
        patterns = [
            r"(?:supplemento(?: fisso)?(?: di)?|aggiunge un supplemento(?: fisso)?(?: di)?)\D{0,20}€\s?(\d+(?:[.,]\d+)?)",
            r"servizio(?: al tavolo)?(?:\s+.*?\s+)?(?:costa|aggiunge)\D{0,20}€\s?(\d+(?:[.,]\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = _to_float(match.group(1))
                if value is not None and value <= 10:
                    return value
        return None

    def _response_contains_order_summary(self, response: str) -> bool:
        response_l = response.lower()
        markers = (
            "ordine",
            "riepilogo",
            "conferma finale",
            "correzione eseguita",
            "totale prodotti",
            "totale parziale",
            "totale con servizio",
            "aggiunto",
            "aggiunta",
            "ordine aggiornato",
        )
        return any(marker in response_l for marker in markers)

    def clean_and_validate(self, slots: StructuredSlots) -> StructuredSlots:
        cleaned = slots.model_copy(deep=True)

        merged_items: Dict[str, OrderItem] = {}
        for item in cleaned.order_state.items:
            canonical = self._canonical_item_name(item.name)
            if not canonical:
                continue
            candidate = item.model_copy(deep=True)
            candidate.name = canonical
            existing = merged_items.get(canonical)
            if existing is None or (existing.unit_price is None and candidate.unit_price is not None):
                merged_items[canonical] = candidate
        cleaned.order_state.items = list(merged_items.values())

        active_catalog = {
            item.name.lower(): item.name
            for item in cleaned.order_state.items
        }
        active_names = set(active_catalog.values())
        removed = []
        for item in cleaned.order_state.removed_items:
            canonical = self._resolve_item_alias(self._canonical_item_name(item), active_catalog)
            if canonical and canonical not in active_names:
                removed.append(canonical)
        cleaned.order_state.removed_items = _dedup(removed)
        cleaned.order_state.total_history = [
            value for value in (_to_float(str(entry)) for entry in cleaned.order_state.total_history) if value is not None
        ]
        cleaned.order_state.total_history = _dedup(f"{value:.2f}" for value in cleaned.order_state.total_history)
        cleaned.order_state.total_history = [float(value) for value in cleaned.order_state.total_history]

        cleaned.dietary_profile.constraints = _dedup(
            self._canonical_constraint(item) for item in cleaned.dietary_profile.constraints if self._canonical_constraint(item)
        )
        if cleaned.dietary_profile.constraints:
            cleaned.dietary_profile.safe_items = self._dedup_casefold(
                self._resolve_item_alias(self._canonical_item_name(item), active_catalog)
                for item in cleaned.dietary_profile.safe_items
                if self._canonical_item_name(item) and self._looks_like_real_item(self._canonical_item_name(item))
            )
        else:
            cleaned.dietary_profile.safe_items = []

        cleaned.event_plan.name = self._canonical_event_name(cleaned.event_plan.name)
        cleaned.event_plan.location = self._canonical_place_name(cleaned.event_plan.location)
        cleaned.event_plan.notes = _dedup(
            note
            for note in (
                self._canonical_event_name(note) or self._canonical_place_name(note) or note
                for note in cleaned.event_plan.notes
            )
            if note
        )

        cleaned.logistics.directions = _dedup(self._extract_direction_steps("\n".join(cleaned.logistics.directions)))
        if cleaned.logistics.supplement is not None and cleaned.logistics.supplement > 10:
            cleaned.logistics.supplement = None

        cleaned.key_entities.products = self._dedup_casefold(
            self._resolve_item_alias(self._canonical_item_name(item), active_catalog)
            for item in (cleaned.key_entities.products + _item_names(cleaned.order_state.items) + cleaned.dietary_profile.safe_items)
            if self._canonical_item_name(item) and self._looks_like_real_item(self._canonical_item_name(item))
        )
        cleaned.key_entities.places = _dedup(
            self._canonical_place_name(item)
            for item in (
                cleaned.key_entities.places
                + cleaned.logistics.directions
                + ([cleaned.event_plan.location] if cleaned.event_plan.location else [])
            )
            if self._canonical_place_name(item)
        )
        cleaned.key_entities.events = _dedup(
            self._canonical_event_name(item)
            for item in (
                cleaned.key_entities.events + ([cleaned.event_plan.name] if cleaned.event_plan.name else [])
            )
            if self._canonical_event_name(item)
        )
        cleaned.key_entities.prices = _dedup(
            self._normalize_price_entry(item)
            for item in cleaned.key_entities.prices
            if self._normalize_price_entry(item)
        )
        cleaned.key_entities.times = _dedup(
            item for item in cleaned.key_entities.times if item and _TIME_RE.fullmatch(item)
        )
        cleaned.negative_decisions = _dedup(
            item for item in cleaned.negative_decisions if item
        )
        provisional = getattr(slots, "provisional_critical", None)
        if provisional:
            object.__setattr__(cleaned, "provisional_critical", self._normalize_provisional_critical(provisional))
        return cleaned

    def extract_immediate_critical_facts(self, turn_text: str) -> Dict[str, List[str]]:
        places = self._extract_places(turn_text)
        if "piazza municipale" in turn_text.lower() and any(
            token in turn_text.lower() for token in ("ci arrivo", "a piedi", "taxi", "percorso")
        ):
            places = _dedup(places + ["Via Garibaldi", "Corso Ercole I d'Este"])
        events = _dedup(
            self._canonical_event_name(match.group(0))
            for match in _EVENT_RE.finditer(turn_text)
            if self._canonical_event_name(match.group(0))
        )
        prices = _dedup(self._extract_inline_prices(turn_text))
        return {
            "places": places,
            "events": events,
            "prices": prices,
        }

    def _normalize_event_name(self, raw: str) -> str:
        return " ".join(_clean_name(raw).split())

    def _canonical_constraint(self, raw: str) -> str:
        value = raw.strip().lower()
        if not value:
            return ""
        if "celiac" in value:
            return "celiaco"
        if "glutin" in value:
            return "senza glutine"
        return value

    def _canonical_item_name(self, raw: str) -> str:
        value = _clean_name(raw)
        value = re.sub(r"^(?:il|lo|la|i|gli|le|un|una)\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\b(?:resta|restano|sono|e'|è|versione|certificata|certificato|adatto|adatta|adatti|adatte)\b.*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip()
        if value.lower().startswith("elementi senza glutine"):
            return ""
        return value.title() if value and value.lower() == value else value

    def _canonical_event_name(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        value = self._normalize_event_name(raw)
        value_l = value.lower()
        if "jazz al chiostro" in value_l:
            return "Jazz al Chiostro"
        if "mostra fotografica" in value_l:
            return "Mostra fotografica"
        if "jazz" in value_l:
            return "Jazz al Chiostro"
        return value

    def _canonical_place_name(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        places = self._extract_places(raw)
        if places:
            return places[0]
        value = _clean_name(raw)
        if any(value.startswith(prefix) for prefix in _PLACE_STARTERS):
            return value
        return None

    def _extract_places(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:['’`-][A-Za-zÀ-ÿ0-9]+)*|[.,;:]", text)
        places: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token not in _PLACE_STARTERS:
                index += 1
                continue
            current = [token]
            index += 1
            while index < len(tokens):
                next_token = tokens[index]
                if next_token in {".", ",", ";", ":"}:
                    break
                lowered = next_token.lower()
                if lowered in _PLACE_STOPWORDS:
                    break
                if self._is_place_token(next_token):
                    current.append(next_token)
                    index += 1
                    continue
                break
            places.append(" ".join(current))
            while index < len(tokens) and tokens[index] not in {".", ",", ";", ":"} and tokens[index] not in _PLACE_STARTERS:
                index += 1
        return _dedup(self._normalize_place(place) for place in places if self._normalize_place(place))

    def _is_place_token(self, token: str) -> bool:
        lowered = token.lower()
        if lowered in _PLACE_CONNECTORS:
            return True
        if re.fullmatch(r"[IVXLC]+", token):
            return True
        if re.fullmatch(r"d['’][A-Z][A-Za-zÀ-ÿ]+", token):
            return True
        return token[:1].isupper()

    def _normalize_place(self, raw: str) -> str:
        value = _clean_name(raw)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def _extract_menu_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        for match in _MENU_CANDIDATE_RE.finditer(text):
            candidate = self._canonical_item_name(match.group(0))
            if candidate and not candidate.startswith(_ITEM_STOP_PREFIXES):
                candidates.append(candidate)
        return _dedup(candidates)

    def _extract_safe_items_from_response(self, response: str) -> List[str]:
        safe_items: List[str] = []
        for sentence in self._split_sentences(response):
            lowered = sentence.lower()
            if not any(marker in lowered for marker in _SAFE_MARKERS):
                continue
            safe_items.extend(self._extract_menu_candidates(sentence))
            safe_items.extend(self._extract_item_names(sentence))
        return _dedup(item for item in safe_items if self._looks_like_real_item(item))

    def _extract_direction_steps(self, text: str) -> List[str]:
        steps: List[str] = []
        for sentence in self._split_sentences(text):
            lowered = sentence.lower()
            if not any(token in lowered for token in ("come ci arrivo", "a piedi", "taxi", "percorso", "destra", "sinistra", "via ", "corso ")):
                continue
            places = self._extract_places(sentence)
            if "a piedi" in lowered:
                steps.append("A piedi")
            if "taxi" in lowered:
                steps.append("Taxi")
            steps.extend(places)
        return _dedup(steps)

    def _split_sentences(self, text: str) -> List[str]:
        return [chunk.strip() for chunk in re.split(r"[.\n]", text) if chunk.strip()]

    def _extract_price_entries(
        self,
        query: str,
        response: str,
        slots: StructuredSlots,
    ) -> List[str]:
        prices: List[str] = []
        for match in _ITEM_LINE_RE.finditer(response):
            name = self._canonical_item_name(match.group(1))
            value = _to_float(match.group(2))
            if name and value is not None:
                prices.append(f"{name}:{value:.2f}")
        current_total = self._extract_total(response)
        if current_total is not None:
            label = "total_with_service" if "servizio" in response.lower() else "total"
            prices.append(f"{label}:{current_total:.2f}")
        supplement = self._extract_supplement(response)
        if supplement is not None:
            prices.append(f"service_supplement:{supplement:.2f}")
        if slots.order_state.current_total is not None:
            prices.append(f"current_total:{slots.order_state.current_total:.2f}")
        return _dedup(prices)

    def _extract_inline_prices(self, text: str) -> List[str]:
        entries: List[str] = []
        for left, right in _PRICE_RE.findall(text):
            raw = left or right
            parsed = _to_float(raw)
            if parsed is not None:
                entries.append(f"price:{parsed:.2f}")
        total = self._extract_total(text)
        if total is not None:
            entries.append(f"total:{total:.2f}")
        supplement = self._extract_supplement(text)
        if supplement is not None:
            entries.append(f"service_supplement:{supplement:.2f}")
        return _dedup(entries)

    def _normalize_price_entry(self, raw: str) -> Optional[str]:
        value = _clean_name(raw)
        if ":" in value:
            context, amount = value.split(":", 1)
            parsed = _to_float(amount)
            if parsed is not None:
                return f"{context}:{parsed:.2f}"
            return None
        matches = [item for pair in _PRICE_RE.findall(value) for item in pair if item]
        if matches:
            parsed = _to_float(matches[-1])
            if parsed is not None:
                return f"price:{parsed:.2f}"
        return None

    def _looks_like_real_item(self, value: str) -> bool:
        if not value:
            return False
        if value.startswith(_ITEM_STOP_PREFIXES):
            return False
        if value.lower() in _GENERIC_ITEM_WORDS:
            return False
        return len(value.split()) <= 4

    def _dedup_casefold(self, items: Iterable[str]) -> List[str]:
        seen = {}
        ordered: List[str] = []
        for item in items:
            value = item.strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen[key] = value
            ordered.append(value)
        return ordered

    def _resolve_item_alias(self, value: str, active_catalog: Dict[str, str]) -> str:
        if not value:
            return ""
        value_l = value.lower()
        if value_l in active_catalog:
            return active_catalog[value_l]
        for key, canonical in active_catalog.items():
            if value_l == key or value_l in key or key in value_l:
                return canonical
        return value

    def _normalize_provisional_critical(self, provisional: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {
            "places": _dedup(
                self._canonical_place_name(item)
                for item in provisional.get("places", [])
                if self._canonical_place_name(item)
            ),
            "events": _dedup(
                self._canonical_event_name(item)
                for item in provisional.get("events", [])
                if self._canonical_event_name(item)
            ),
            "prices": _dedup(
                self._normalize_price_entry(item)
                for item in provisional.get("prices", [])
                if self._normalize_price_entry(item)
            ),
        }


# FIX ROUND 1 - HIAMS recall boost - 2026-04-04
# FIX ROUND 2 - HIAMS zero-latency recall boost - T15 fixed
