"""
Heuristic Engine - Multi-layered signal extraction system.

This is the core engine that processes queries through all 4 layers:
- Layer A: Hard patterns (binary flags)
- Layer B: Keyword scoring (weighted capabilities)
- Layer C: Fuzzy matching (controlled, whitelist-based)
- Layer D: Negation rules (score subtraction/zeroing)

The engine produces PatternSignals, NOT routing decisions.
The router uses these signals to make the final decision.

v2.0 CHANGES:
- Enhanced explicit_source handling with web/internal type distinction
- Support for negations in Layer B patterns
- Improved pattern loading for new YAML structure
- Better commercial intent detection

FIX-011 v1.8.2:
- Added Prometheus metrics integration
- Metrics recorded for each query analysis

ROADMAP v1.7.x - FEAT-ROUTER-002 / FEAT-ROUTER-003
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass

import yaml

# Import env expansion for 12-Factor YAML configuration
from ubp_enterprise_hybrid.modules.cores._shared.manifest_loader import expand_env_vars

from .models import (
    PatternSignals,
    HardFlags,
    CapabilityScores,
    TraceEntry,
    PatternUnit,
    NegationRule,
)
from .fuzzy import FuzzyMatcher, get_fuzzy_matcher

# FIX-011 v1.8.2: Prometheus metrics
from .metrics import record_query_metrics, record_layer_hits, set_pattern_counts

logger = logging.getLogger(__name__)


def _load_yaml_with_env(path: Path) -> Dict[str, Any]:
    """
    Load YAML file with environment variable expansion.

    Supports ${VAR} and ${VAR:-default} syntax for 12-Factor compliance.
    This allows tuning pattern weights via environment variables.

    Example in YAML:
        weight: "${UBP_ROUTER_FRESHNESS_WEIGHT:-1.5}"

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML data with environment variables expanded
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Expand ${VAR} and ${VAR:-default} placeholders
    expanded_content = expand_env_vars(raw_content)

    # Parse YAML
    data = yaml.safe_load(expanded_content)

    # Coerce string booleans and numbers (similar to JSON coercion)
    def coerce_values(obj):
        if isinstance(obj, dict):
            return {k: coerce_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [coerce_values(item) for item in obj]
        elif isinstance(obj, str):
            lower = obj.lower()
            if lower == "true":
                return True
            elif lower == "false":
                return False
            # Try numeric conversion
            try:
                if "." in obj:
                    return float(obj)
                return int(obj)
            except ValueError:
                return obj
        return obj

    return coerce_values(data) if data else {}


@dataclass
class EngineConfig:
    """Configuration for the heuristic engine."""

    config_dir: Path
    enable_fuzzy: bool = True
    enable_negations: bool = True
    normalize_scores: bool = True
    max_score: float = 1.0
    debug_mode: bool = False


class HeuristicEngine:
    """
    Multi-layered heuristic signal extraction engine.

    Processes queries through 4 layers to extract signals:
    - Layer A: Hard patterns → Binary flags
    - Layer B: Keyword scoring → Capability scores
    - Layer C: Fuzzy matching → Score boost for typos
    - Layer D: Negations → Score subtraction

    Output is PatternSignals, not a routing decision.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize the heuristic engine.

        Args:
            config: Engine configuration. If None, uses defaults.
        """
        self.config = config or EngineConfig(
            config_dir=Path(__file__).parent / "config"
        )

        # Layer A: Hard patterns
        self.hard_patterns: Dict[str, Any] = {}
        self._compiled_hard_patterns: Dict[str, List[re.Pattern]] = {}
        # v2.0: Explicit source keywords for fast lookup
        self._explicit_source_web_keywords: Set[str] = set()
        self._explicit_source_internal_keywords: Set[str] = set()

        # Layer B: Keyword patterns
        self.pattern_units: List[PatternUnit] = []
        self._token_to_patterns: Dict[str, List[PatternUnit]] = {}

        # Layer C: Fuzzy matcher
        self.fuzzy_matcher: Optional[FuzzyMatcher] = None

        # Layer D: Negation rules
        self.negation_rules: List[NegationRule] = []

        # Statistics
        self.total_analyses = 0
        self.layer_hits = {"A": 0, "B": 0, "C": 0, "D": 0}

        # Load configurations
        self._load_all_configs()

        logger.info(
            "HeuristicEngine initialized",
            extra={
                "hard_patterns": len(self.hard_patterns),
                "pattern_units": len(self.pattern_units),
                "negation_rules": len(self.negation_rules),
                "fuzzy_enabled": self.config.enable_fuzzy,
            },
        )

    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        config_dir = self.config.config_dir

        # Load Layer A: Hard patterns
        hard_patterns_path = config_dir / "hard_patterns.yaml"
        if hard_patterns_path.exists():
            self._load_hard_patterns(hard_patterns_path)

        # Load Layer B: Keyword patterns
        patterns_path = config_dir / "patterns.yaml"
        if patterns_path.exists():
            self._load_patterns(patterns_path)

        # Load Layer C: Fuzzy whitelist
        if self.config.enable_fuzzy:
            fuzzy_path = config_dir / "fuzzy_whitelist.yaml"
            if fuzzy_path.exists():
                self.fuzzy_matcher = FuzzyMatcher(fuzzy_path)
            else:
                self.fuzzy_matcher = FuzzyMatcher()

        # Load Layer D: Negations
        if self.config.enable_negations:
            negations_path = config_dir / "negations.yaml"
            if negations_path.exists():
                self._load_negations(negations_path)

        # FIX-011 v1.8.2: Set pattern count metrics for monitoring
        try:
            hard_count = sum(len(p) for p in self._compiled_hard_patterns.values())
            soft_count = len(self.pattern_units)
            negation_count = len(self.negation_rules)
            fuzzy_count = len(self.fuzzy_matcher.whitelist) if self.fuzzy_matcher else 0
            set_pattern_counts(hard_count, soft_count, negation_count, fuzzy_count)
        except Exception as e:
            logger.warning(f"Pattern metrics setup failed (non-critical): {e}")

    def _load_hard_patterns(self, path: Path) -> None:
        """Load Layer A hard patterns from YAML with env expansion."""
        try:
            # Use env-aware YAML loader (12-Factor compliant)
            self.hard_patterns = _load_yaml_with_env(path)

            # Pre-compile regex patterns
            for category, data in self.hard_patterns.items():
                if category == "version":
                    continue

                self._compiled_hard_patterns[category] = []

                if isinstance(data, dict):
                    # v2.0: Handle new explicit_source structure with web_patterns/internal_patterns
                    if category == "explicit_source":
                        # Compile web patterns
                        web_patterns = data.get("web_patterns", [])
                        for p in web_patterns:
                            self._compile_and_store_pattern(p, "explicit_source_web")

                        # Compile internal patterns
                        internal_patterns = data.get("internal_patterns", [])
                        for p in internal_patterns:
                            self._compile_and_store_pattern(
                                p, "explicit_source_internal"
                            )

                        # Store keywords for fast lookup
                        self._explicit_source_web_keywords = set(
                            kw.lower() for kw in data.get("web_keywords", [])
                        )
                        self._explicit_source_internal_keywords = set(
                            kw.lower() for kw in data.get("internal_keywords", [])
                        )
                        continue

                    # Standard pattern loading
                    patterns = data.get("patterns", [])
                    for p in patterns:
                        self._compile_and_store_pattern(p, category)

            logger.info(
                f"Loaded {len(self._compiled_hard_patterns)} hard pattern categories"
            )

        except Exception as e:
            logger.error(f"Failed to load hard patterns: {e}")

    def _compile_and_store_pattern(self, p: Any, category: str) -> None:
        """Helper to compile and store a regex pattern."""
        if category not in self._compiled_hard_patterns:
            self._compiled_hard_patterns[category] = []

        if isinstance(p, dict):
            pattern_str = p.get("pattern", "")
            flags_str = p.get("flags", "")
            flags = re.IGNORECASE if "i" in flags_str else 0
        else:
            pattern_str = p
            flags = re.IGNORECASE

        try:
            compiled = re.compile(pattern_str, flags)
            self._compiled_hard_patterns[category].append(compiled)
        except re.error as e:
            logger.warning(f"Invalid pattern in {category}: {pattern_str} - {e}")

    def _load_patterns(self, path: Path) -> None:
        """Load Layer B keyword patterns from YAML with env expansion."""
        try:
            # Use env-aware YAML loader (12-Factor compliant)
            config = _load_yaml_with_env(path)

            defaults = config.get("defaults", {})
            default_weight = defaults.get("weight", 1.0)
            default_boost_mult = defaults.get("boost_multiplier", 1.3)

            # Process each category
            for category, patterns in config.items():
                if category in ("version", "defaults"):
                    continue

                if not isinstance(patterns, list):
                    continue

                for p in patterns:
                    unit = PatternUnit(
                        id=p.get("id", ""),
                        tokens=[t.lower() for t in p.get("tokens", [])],
                        weight=p.get("weight", default_weight),
                        scope=p.get("scope", []),
                        negations=p.get("negations", []),
                        boosts=[b.lower() for b in p.get("boosts", [])],
                        boost_multiplier=p.get("boost_multiplier", default_boost_mult),
                        description=p.get("description", ""),
                    )

                    self.pattern_units.append(unit)

                    # Build token index for fast lookup
                    for token in unit.tokens:
                        if token not in self._token_to_patterns:
                            self._token_to_patterns[token] = []
                        self._token_to_patterns[token].append(unit)

            logger.info(f"Loaded {len(self.pattern_units)} pattern units")

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")

    def _load_negations(self, path: Path) -> None:
        """Load Layer D negation rules from YAML with env expansion."""
        try:
            # Use env-aware YAML loader (12-Factor compliant)
            config = _load_yaml_with_env(path)

            for category, rules in config.items():
                if category == "version":
                    continue

                if not isinstance(rules, list):
                    continue

                for r in rules:
                    rule = NegationRule(
                        triggers=[t.lower() for t in r.get("triggers", [])],
                        affects=r.get("affects", []),
                        effect=r.get("effect", "subtract"),
                        value=r.get("value", 0.5),
                        constraint=r.get("constraint"),
                    )
                    self.negation_rules.append(rule)

            logger.info(f"Loaded {len(self.negation_rules)} negation rules")

        except Exception as e:
            logger.error(f"Failed to load negations: {e}")

    def analyze(self, query: str) -> PatternSignals:
        """
        Analyze query through all layers.

        Args:
            query: User query to analyze

        Returns:
            PatternSignals with all extracted signals
        """
        start_time = time.perf_counter()
        self.total_analyses += 1

        # Initialize result
        signals = PatternSignals(
            query=query,
            flags=HardFlags(),
            capabilities=CapabilityScores(),
        )

        # Normalize query
        query_normalized = query.strip()
        query_lower = query_normalized.lower()
        tokens = self._tokenize(query_lower)
        signals.tokens_analyzed = len(tokens)

        # === Layer A: Hard patterns (binary flags) ===
        self._apply_layer_a(query_normalized, query_lower, signals)

        # === Layer B: Keyword scoring ===
        self._apply_layer_b(query_lower, tokens, signals)

        # === Layer C: Fuzzy matching ===
        if self.config.enable_fuzzy and self.fuzzy_matcher:
            self._apply_layer_c(tokens, signals)

        # === Layer D: Negations ===
        if self.config.enable_negations:
            self._apply_layer_d(query_lower, signals)

        # Normalize scores
        if self.config.normalize_scores:
            signals.capabilities.normalize(self.config.max_score)

        # Calculate ambiguity
        signals.ambiguity = self._calculate_ambiguity(signals)

        # Processing time
        signals.processing_time_ms = (time.perf_counter() - start_time) * 1000

        if self.config.debug_mode:
            logger.debug(
                f"Query analyzed",
                extra={
                    "query": query[:50],
                    "signals": signals.to_dict(),
                },
            )

        # FIX-011 v1.8.2: Record Prometheus metrics
        try:
            record_query_metrics(signals)
            record_layer_hits(self.layer_hits)
        except Exception as e:
            logger.warning(f"Metrics recording failed (non-critical): {e}")

        return signals

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for pattern matching.

        Simple tokenization - splits on whitespace and punctuation.
        """
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        tokens = text.split()
        return [t.strip("'") for t in tokens if t.strip("'")]

    def _apply_layer_a(
        self, query: str, query_lower: str, signals: PatternSignals
    ) -> None:
        """Apply Layer A: Hard patterns for binary flags."""

        # Check greetings
        if "greetings" in self.hard_patterns:
            greetings = self.hard_patterns["greetings"]
            exact_greetings = set(g.lower() for g in greetings.get("exact", []))

            # Check exact match (trimmed, single word)
            query_trimmed = query_lower.strip("!?., ")
            if query_trimmed in exact_greetings:
                signals.flags.greeting_detected = True
                signals.add_trace(
                    query_trimmed, 1.0, "A", "greeting", "Exact greeting match"
                )
                self.layer_hits["A"] += 1

            # Check pattern match
            for pattern in self._compiled_hard_patterns.get("greetings", []):
                if pattern.match(query):
                    signals.flags.greeting_detected = True
                    signals.add_trace(
                        pattern.pattern, 1.0, "A", "greeting", "Pattern greeting match"
                    )
                    self.layer_hits["A"] += 1
                    break

        # Check farewells
        if "farewells" in self.hard_patterns:
            farewells = self.hard_patterns["farewells"]
            exact_farewells = set(f.lower() for f in farewells.get("exact", []))

            query_trimmed = query_lower.strip("!?., ")
            if query_trimmed in exact_farewells:
                signals.flags.farewell_detected = True
                signals.add_trace(
                    query_trimmed, 1.0, "A", "farewell", "Exact farewell match"
                )
                self.layer_hits["A"] += 1

            for pattern in self._compiled_hard_patterns.get("farewells", []):
                if pattern.match(query):
                    signals.flags.farewell_detected = True
                    signals.add_trace(
                        pattern.pattern, 1.0, "A", "farewell", "Pattern farewell match"
                    )
                    self.layer_hits["A"] += 1
                    break

        # Check acknowledgments (treat as chat/greeting)
        if "acknowledgments" in self.hard_patterns:
            acks = self.hard_patterns["acknowledgments"]
            exact_acks = set(a.lower() for a in acks.get("exact", []))

            query_trimmed = query_lower.strip("!?., ")
            if query_trimmed in exact_acks:
                signals.flags.greeting_detected = True  # Treat as conversational
                signals.add_trace(
                    query_trimmed, 1.0, "A", "acknowledgment", "Acknowledgment match"
                )
                self.layer_hits["A"] += 1

            for pattern in self._compiled_hard_patterns.get("acknowledgments", []):
                if pattern.match(query):
                    signals.flags.greeting_detected = True
                    signals.add_trace(
                        pattern.pattern,
                        1.0,
                        "A",
                        "acknowledgment",
                        "Pattern acknowledgment",
                    )
                    self.layer_hits["A"] += 1
                    break

        # Check temporal patterns
        for pattern in self._compiled_hard_patterns.get("temporal", []):
            if pattern.search(query_lower):
                signals.flags.temporal_query = True
                signals.add_trace(
                    pattern.pattern, 1.0, "A", "temporal", "Temporal marker detected"
                )
                self.layer_hits["A"] += 1
                break

        # Check internal reference patterns
        if "internal_reference" in self.hard_patterns:
            int_ref = self.hard_patterns["internal_reference"]
            keywords = [k.lower() for k in int_ref.get("keywords", [])]

            for kw in keywords:
                if kw in query_lower:
                    signals.flags.internal_reference = True
                    signals.add_trace(
                        kw, 1.0, "A", "internal", "Internal reference keyword"
                    )
                    self.layer_hits["A"] += 1
                    break

            if not signals.flags.internal_reference:
                for pattern in self._compiled_hard_patterns.get(
                    "internal_reference", []
                ):
                    if pattern.search(query_lower):
                        signals.flags.internal_reference = True
                        signals.add_trace(
                            pattern.pattern,
                            1.0,
                            "A",
                            "internal",
                            "Internal reference pattern",
                        )
                        self.layer_hits["A"] += 1
                        break

        # Check policy sensitive
        if "policy_sensitive" in self.hard_patterns:
            policy = self.hard_patterns["policy_sensitive"]
            keywords = [k.lower() for k in policy.get("keywords", [])]

            for kw in keywords:
                if kw in query_lower:
                    signals.flags.policy_sensitive = True
                    signals.add_trace(
                        kw, 1.0, "A", "policy", "Policy-sensitive keyword"
                    )
                    self.layer_hits["A"] += 1
                    break

        # Check safety
        if "safety" in self.hard_patterns:
            safety = self.hard_patterns["safety"]
            keywords = [k.lower() for k in safety.get("keywords", [])]

            for kw in keywords:
                if kw in query_lower:
                    signals.flags.safety_triggered = True
                    signals.add_trace(kw, 1.0, "A", "safety", "Safety keyword detected")
                    self.layer_hits["A"] += 1
                    break

        # Check explicit source (v2.0: with web/internal type distinction)
        # First check keywords (fast path)
        for kw in self._explicit_source_web_keywords:
            if kw in query_lower:
                signals.flags.explicit_source = True
                signals.flags.explicit_source_type = "web"
                signals.add_trace(
                    kw, 1.0, "A", "explicit_source", "Explicit WEB source keyword"
                )
                self.layer_hits["A"] += 1
                break

        if not signals.flags.explicit_source:
            for kw in self._explicit_source_internal_keywords:
                if kw in query_lower:
                    signals.flags.explicit_source = True
                    signals.flags.explicit_source_type = "internal"
                    signals.add_trace(
                        kw,
                        1.0,
                        "A",
                        "explicit_source",
                        "Explicit INTERNAL source keyword",
                    )
                    self.layer_hits["A"] += 1
                    break

        # Then check regex patterns
        if not signals.flags.explicit_source:
            for pattern in self._compiled_hard_patterns.get("explicit_source_web", []):
                if pattern.search(query_lower):
                    signals.flags.explicit_source = True
                    signals.flags.explicit_source_type = "web"
                    signals.add_trace(
                        pattern.pattern,
                        1.0,
                        "A",
                        "explicit_source",
                        "Explicit WEB source pattern",
                    )
                    self.layer_hits["A"] += 1
                    break

        if not signals.flags.explicit_source:
            for pattern in self._compiled_hard_patterns.get(
                "explicit_source_internal", []
            ):
                if pattern.search(query_lower):
                    signals.flags.explicit_source = True
                    signals.flags.explicit_source_type = "internal"
                    signals.add_trace(
                        pattern.pattern,
                        1.0,
                        "A",
                        "explicit_source",
                        "Explicit INTERNAL source pattern",
                    )
                    self.layer_hits["A"] += 1
                    break

    def _apply_layer_b(
        self, query_lower: str, tokens: List[str], signals: PatternSignals
    ) -> None:
        """Apply Layer B: Keyword scoring with negation support."""

        matched_patterns: Set[str] = set()

        for token in tokens:
            # Check if token matches any pattern
            if token in self._token_to_patterns:
                for unit in self._token_to_patterns[token]:
                    if unit.id in matched_patterns:
                        continue

                    # v2.0: Check for negation words that block this pattern
                    negated = False
                    if unit.negations:
                        for neg_word in unit.negations:
                            if neg_word in query_lower:
                                negated = True
                                signals.add_trace(
                                    pattern=token,
                                    effect=0.0,
                                    layer="B",
                                    scope=",".join(unit.scope),
                                    reason=f"Blocked by negation '{neg_word}'",
                                )
                                break

                    if negated:
                        matched_patterns.add(
                            unit.id
                        )  # Mark as matched to avoid reprocessing
                        continue

                    matched_patterns.add(unit.id)

                    # Calculate score with potential boost
                    score = unit.weight
                    boost_applied = False

                    # Check for boost words in query
                    for boost in unit.boosts:
                        if boost in query_lower:
                            score *= unit.boost_multiplier
                            boost_applied = True
                            break

                    # Apply score to relevant capabilities
                    for scope in unit.scope:
                        if scope == "needs_fresh_info":
                            signals.capabilities.needs_fresh_info += score
                        elif scope == "internal_knowledge":
                            signals.capabilities.internal_knowledge += score
                        elif scope == "pure_conversation":
                            signals.capabilities.pure_conversation += score
                        elif scope == "commercial_info":
                            signals.capabilities.commercial_info += score
                        elif scope == "technical_depth":
                            signals.capabilities.technical_depth += score

                    reason = f"Matched '{token}'"
                    if boost_applied:
                        reason += " (boosted)"

                    signals.add_trace(
                        pattern=token,
                        effect=score,
                        layer="B",
                        scope=",".join(unit.scope),
                        reason=reason,
                    )
                    self.layer_hits["B"] += 1

        # Also check multi-token patterns
        for unit in self.pattern_units:
            if unit.id in matched_patterns:
                continue

            for token in unit.tokens:
                if " " in token and token in query_lower:
                    # v2.0: Check for negation words
                    negated = False
                    if unit.negations:
                        for neg_word in unit.negations:
                            if neg_word in query_lower:
                                negated = True
                                break

                    if negated:
                        matched_patterns.add(unit.id)
                        break

                    matched_patterns.add(unit.id)

                    score = unit.weight

                    for boost in unit.boosts:
                        if boost in query_lower:
                            score *= unit.boost_multiplier
                            break

                    for scope in unit.scope:
                        if scope == "needs_fresh_info":
                            signals.capabilities.needs_fresh_info += score
                        elif scope == "internal_knowledge":
                            signals.capabilities.internal_knowledge += score
                        elif scope == "pure_conversation":
                            signals.capabilities.pure_conversation += score
                        elif scope == "commercial_info":
                            signals.capabilities.commercial_info += score
                        elif scope == "technical_depth":
                            signals.capabilities.technical_depth += score

                    signals.add_trace(
                        pattern=token,
                        effect=score,
                        layer="B",
                        scope=",".join(unit.scope),
                        reason=f"Multi-token match",
                    )
                    self.layer_hits["B"] += 1
                    break

    def _apply_layer_c(self, tokens: List[str], signals: PatternSignals) -> None:
        """Apply Layer C: Fuzzy matching."""

        if not self.fuzzy_matcher:
            return

        fuzzy_matches = self.fuzzy_matcher.match_tokens(tokens)

        for original, match in fuzzy_matches.items():
            # Check if the matched canonical word is in our patterns
            if match.matched in self._token_to_patterns:
                for unit in self._token_to_patterns[match.matched]:
                    # Apply reduced score based on fuzzy confidence
                    score = (
                        unit.weight * match.confidence * 0.8
                    )  # 20% penalty for fuzzy

                    for scope in unit.scope:
                        if scope == "needs_fresh_info":
                            signals.capabilities.needs_fresh_info += score
                        elif scope == "internal_knowledge":
                            signals.capabilities.internal_knowledge += score
                        elif scope == "pure_conversation":
                            signals.capabilities.pure_conversation += score
                        elif scope == "commercial_info":
                            signals.capabilities.commercial_info += score
                        elif scope == "technical_depth":
                            signals.capabilities.technical_depth += score

                    signals.add_trace(
                        pattern=f"{original}→{match.matched}",
                        effect=score,
                        layer="C",
                        scope=",".join(unit.scope),
                        reason=f"Fuzzy match (dist={match.distance}, conf={match.confidence:.2f})",
                    )
                    self.layer_hits["C"] += 1

    def _apply_layer_d(self, query_lower: str, signals: PatternSignals) -> None:
        """Apply Layer D: Negation rules.

        FIX-004 v1.8.2: GUARD - Skip modifications if Layer A set explicit_source flag.
        This prevents negations from overriding explicit user intent (e.g., "cerca online").
        """
        # === GUARD: Respect Layer A hard flags ===
        if signals.flags.explicit_source:
            logger.debug(
                "Layer D: Skipping negations - explicit_source set by Layer A",
                extra={"source_type": signals.flags.explicit_source_type}
            )
            return

        for rule in self.negation_rules:
            for trigger in rule.triggers:
                if trigger in query_lower:
                    # Apply negation effect
                    for affected in rule.affects:
                        if rule.effect == "zero":
                            if affected == "needs_fresh_info":
                                signals.capabilities.needs_fresh_info = 0.0
                            elif affected == "internal_knowledge":
                                signals.capabilities.internal_knowledge = 0.0
                            elif affected == "pure_conversation":
                                signals.capabilities.pure_conversation = 0.0
                            elif affected == "commercial_info":
                                signals.capabilities.commercial_info = 0.0
                            elif affected == "technical_depth":
                                signals.capabilities.technical_depth = 0.0

                        elif rule.effect == "subtract":
                            if affected == "needs_fresh_info":
                                signals.capabilities.needs_fresh_info -= rule.value
                            elif affected == "internal_knowledge":
                                signals.capabilities.internal_knowledge -= rule.value
                            elif affected == "pure_conversation":
                                signals.capabilities.pure_conversation -= rule.value
                            elif affected == "commercial_info":
                                signals.capabilities.commercial_info -= rule.value
                            elif affected == "technical_depth":
                                signals.capabilities.technical_depth -= rule.value

                    signals.negations_applied.append(trigger)

                    effect_desc = f"{rule.effect}"
                    if rule.effect == "subtract":
                        effect_desc += f"({rule.value})"

                    signals.add_trace(
                        pattern=trigger,
                        effect=-rule.value if rule.effect == "subtract" else -1.0,
                        layer="D",
                        scope=",".join(rule.affects),
                        reason=f"Negation: {effect_desc}",
                    )
                    self.layer_hits["D"] += 1
                    break  # Only apply first matching trigger per rule

        # Ensure no negative scores
        signals.capabilities.needs_fresh_info = max(
            0.0, signals.capabilities.needs_fresh_info
        )
        signals.capabilities.internal_knowledge = max(
            0.0, signals.capabilities.internal_knowledge
        )
        signals.capabilities.pure_conversation = max(
            0.0, signals.capabilities.pure_conversation
        )
        signals.capabilities.commercial_info = max(
            0.0, signals.capabilities.commercial_info
        )
        signals.capabilities.technical_depth = max(
            0.0, signals.capabilities.technical_depth
        )

    def _calculate_ambiguity(self, signals: PatternSignals) -> float:
        """
        Calculate ambiguity score based on signals.

        Low ambiguity if:
        - Hard flags are set
        - One capability clearly dominates

        High ambiguity if:
        - No hard flags
        - Multiple capabilities have similar scores
        - Low overall signal strength
        """
        # Hard flags = low ambiguity
        if signals.flags.has_any_flag():
            return 0.1

        cap_scores = list(signals.capabilities.to_dict().values())

        # No signals = high ambiguity
        if not cap_scores or max(cap_scores) == 0:
            return 0.9

        max_score = max(cap_scores)
        second_max = sorted(cap_scores, reverse=True)[1] if len(cap_scores) > 1 else 0

        # Calculate ambiguity from score distribution
        if max_score > 0:
            gap_ratio = (max_score - second_max) / max_score
            ambiguity = 1.0 - gap_ratio
        else:
            ambiguity = 1.0

        # Low absolute scores = more ambiguity
        if max_score < 0.5:
            ambiguity = min(1.0, ambiguity + 0.2)

        return min(max(ambiguity, 0.0), 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_analyses": self.total_analyses,
            "layer_hits": self.layer_hits,
            "pattern_units_loaded": len(self.pattern_units),
            "negation_rules_loaded": len(self.negation_rules),
            "hard_pattern_categories": len(self._compiled_hard_patterns),
            "fuzzy_enabled": self.fuzzy_matcher is not None,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_analyses = 0
        self.layer_hits = {"A": 0, "B": 0, "C": 0, "D": 0}


# Singleton instance
_engine: Optional[HeuristicEngine] = None


def get_heuristic_engine(config: Optional[EngineConfig] = None) -> HeuristicEngine:
    """Get or create the singleton HeuristicEngine instance."""
    global _engine

    if _engine is None:
        _engine = HeuristicEngine(config)

    return _engine


__all__ = [
    "HeuristicEngine",
    "EngineConfig",
    "get_heuristic_engine",
]
