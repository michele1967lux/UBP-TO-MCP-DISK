"""
Layer C - Fuzzy Matching Module

Controlled fuzzy matching for whitelisted keywords only.
Never applies to: legal terms, codes, numbers, versions.

ROADMAP v1.7.x - FEAT-ROUTER-002
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import yaml

from .models import FuzzyMatch

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Optimized for short strings (typical in our use case).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class FuzzyMatcher:
    """
    Controlled fuzzy matching engine.

    Only matches against whitelisted words with strict distance limits.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize fuzzy matcher.

        Args:
            config_path: Path to fuzzy_whitelist.yaml config
        """
        self.enabled = True
        self.max_distance_by_length: Dict[int, int] = {
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
        }
        self.default_max_distance = 1
        self.min_confidence = 0.7

        # FIX-009 v1.8.2: Bounded whitelist to prevent memory leaks
        self.max_whitelist_size = 10000  # Maximum canonical words allowed
        self.max_variants_per_word = 50  # Maximum variants per canonical word

        # Whitelist: canonical_word -> set of known variants
        self.whitelist: Dict[str, Set[str]] = {}

        # Reverse index: variant -> canonical_word
        self.variant_to_canonical: Dict[str, str] = {}

        # Blacklist patterns (compiled regex)
        self.blacklist_patterns: List[re.Pattern] = []
        self.blacklist_words: Set[str] = set()

        if config_path:
            self._load_config(config_path)

        logger.info(
            f"FuzzyMatcher initialized",
            extra={
                "enabled": self.enabled,
                "whitelist_size": len(self.whitelist),
                "blacklist_patterns": len(self.blacklist_patterns),
            },
        )

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Settings
            settings = config.get("settings", {})
            self.enabled = settings.get("enabled", True)
            self.max_distance_by_length = settings.get(
                "max_distance_by_length", self.max_distance_by_length
            )
            self.default_max_distance = settings.get("default_max_distance", 1)
            self.min_confidence = settings.get("min_confidence", 0.7)

            # Build whitelist
            whitelist_config = config.get("whitelist", {})
            for category, words in whitelist_config.items():
                for canonical, variants in words.items():
                    canonical_lower = canonical.lower()
                    self.whitelist[canonical_lower] = set(v.lower() for v in variants)
                    self.whitelist[canonical_lower].add(canonical_lower)

                    # Build reverse index
                    for variant in self.whitelist[canonical_lower]:
                        self.variant_to_canonical[variant] = canonical_lower

            # Build blacklist
            blacklist_config = config.get("blacklist", {})

            for pattern in blacklist_config.get("patterns", []):
                try:
                    self.blacklist_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid blacklist pattern '{pattern}': {e}")

            self.blacklist_words = set(
                w.lower() for w in blacklist_config.get("words", [])
            )

            logger.info(f"Loaded fuzzy config from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load fuzzy config: {e}")

    def _get_max_distance(self, word_length: int) -> int:
        """Get maximum allowed edit distance for word length."""
        # Check specific lengths first
        for length, max_dist in sorted(self.max_distance_by_length.items()):
            if word_length <= length:
                return max_dist
        return self.default_max_distance

    def _is_blacklisted(self, word: str) -> bool:
        """Check if word is blacklisted from fuzzy matching."""
        word_lower = word.lower()

        # Check explicit blacklist words
        if word_lower in self.blacklist_words:
            return True

        # Check blacklist patterns
        for pattern in self.blacklist_patterns:
            if pattern.match(word):
                return True

        return False

    def match(self, token: str) -> Optional[FuzzyMatch]:
        """
        Attempt to fuzzy match a token against whitelist.

        Args:
            token: Token to match

        Returns:
            FuzzyMatch if successful, None otherwise
        """
        if not self.enabled:
            return None

        token_lower = token.lower()

        # Check if blacklisted
        if self._is_blacklisted(token):
            return None

        # Check if it's already a known variant (exact match)
        if token_lower in self.variant_to_canonical:
            canonical = self.variant_to_canonical[token_lower]
            return FuzzyMatch(
                original=token, matched=canonical, distance=0, confidence=1.0
            )

        # Attempt fuzzy matching against all canonical words
        max_distance = self._get_max_distance(len(token))
        best_match: Optional[FuzzyMatch] = None
        best_distance = float("inf")

        for canonical, variants in self.whitelist.items():
            # Check against canonical form
            distance = levenshtein_distance(token_lower, canonical)

            if distance <= max_distance and distance < best_distance:
                confidence = 1.0 - (distance / max(len(token_lower), len(canonical)))

                if confidence >= self.min_confidence:
                    best_match = FuzzyMatch(
                        original=token,
                        matched=canonical,
                        distance=distance,
                        confidence=confidence,
                    )
                    best_distance = distance

            # Also check against known variants
            for variant in variants:
                distance = levenshtein_distance(token_lower, variant)

                if distance <= max_distance and distance < best_distance:
                    confidence = 1.0 - (distance / max(len(token_lower), len(variant)))

                    if confidence >= self.min_confidence:
                        best_match = FuzzyMatch(
                            original=token,
                            matched=canonical,  # Return canonical, not variant
                            distance=distance,
                            confidence=confidence,
                        )
                        best_distance = distance

        return best_match

    def match_tokens(self, tokens: List[str]) -> Dict[str, FuzzyMatch]:
        """
        Batch match multiple tokens.

        Args:
            tokens: List of tokens to match

        Returns:
            Dict mapping original token to FuzzyMatch (only for matches)
        """
        results = {}

        for token in tokens:
            match = self.match(token)
            if match and match.distance > 0:  # Only include actual fuzzy matches
                results[token] = match

        return results

    def add_to_whitelist(
        self, canonical: str, variants: List[str], category: str = "dynamic"
    ) -> bool:
        """
        Dynamically add words to whitelist with bounds checking.

        FIX-009 v1.8.2: Added size limits to prevent memory leaks.

        Args:
            canonical: The canonical form of the word
            variants: List of accepted variants/typos
            category: Category name for logging

        Returns:
            True if added successfully, False if bounds exceeded
        """
        canonical_lower = canonical.lower()

        # FIX-009: Check whitelist size bounds
        if canonical_lower not in self.whitelist:
            if len(self.whitelist) >= self.max_whitelist_size:
                logger.warning(
                    f"Whitelist size limit reached ({self.max_whitelist_size}), "
                    f"rejecting new word: {canonical}"
                )
                return False
            self.whitelist[canonical_lower] = set()

        # FIX-009: Limit variants per word
        current_variants = self.whitelist[canonical_lower]
        new_variants = {v.lower() for v in variants}
        new_variants.add(canonical_lower)

        # Calculate how many we can add
        space_available = self.max_variants_per_word - len(current_variants)
        if space_available <= 0:
            logger.debug(
                f"Variant limit reached for '{canonical}' ({self.max_variants_per_word})"
            )
            return True  # Word exists, just can't add more variants

        # Add only what fits
        variants_to_add = list(new_variants - current_variants)[:space_available]
        current_variants.update(variants_to_add)

        # Update reverse index for new variants only
        for variant in variants_to_add:
            self.variant_to_canonical[variant] = canonical_lower

        logger.debug(f"Added '{canonical}' to fuzzy whitelist ({category})")
        return True

    def get_stats(self) -> Dict[str, int]:
        """Get matcher statistics."""
        return {
            "enabled": self.enabled,
            "whitelist_canonical_count": len(self.whitelist),
            "whitelist_total_variants": sum(len(v) for v in self.whitelist.values()),
            "blacklist_patterns_count": len(self.blacklist_patterns),
            "blacklist_words_count": len(self.blacklist_words),
        }


# Singleton instance (lazy initialization)
_fuzzy_matcher: Optional[FuzzyMatcher] = None


def get_fuzzy_matcher(config_path: Optional[Path] = None) -> FuzzyMatcher:
    """Get or create the singleton FuzzyMatcher instance."""
    global _fuzzy_matcher

    if _fuzzy_matcher is None:
        _fuzzy_matcher = FuzzyMatcher(config_path)

    return _fuzzy_matcher


__all__ = [
    "FuzzyMatcher",
    "FuzzyMatch",
    "levenshtein_distance",
    "get_fuzzy_matcher",
]
